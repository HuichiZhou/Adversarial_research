import torch
import torch.nn.functional as F
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])
    

def add_hooks(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 50257: # 50257 which is the GPT2's vocabulary
                module.weight.requires_grad = True
                module.register_backward_hook(extract_grad_hook)

def get_embedding_weight(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 50257:
                return module.weight.detach()
            
def make_target_batch(tokenizer, device, target_texts):
    encoded_texts = []
    max_len = 0
    
    for target_text in target_texts:
        encoded_target_text = tokenizer.encode(target_text) # 对target_text进行编码
        encoded_texts.append(encoded_target_text) # 将编码后的target_text加入到encoded_texts中
        if len(encoded_target_text) > max_len:          
            max_len = len(encoded_target_text)  # 记录最长的target_text的长度

    for indx, encoded_text in enumerate(encoded_texts):     
        if len(encoded_text) < max_len:     
            encoded_texts[indx].extend([-1] * (max_len - len(encoded_text)))    # 将长度不足max_len的target_text补齐

    target_tokens_batch = None  
    for encoded_text in encoded_texts:  
        target_tokens = torch.tensor(encoded_text, device=device, dtype=torch.long).unsqueeze(0)    # 将target_text转换为tensor
        if target_tokens_batch is None:
            target_tokens_batch = target_tokens
        else:
            target_tokens_batch = torch.cat((target_tokens, target_tokens_batch), dim=0)    # 将target_tokens_batch拼接起来
    return target_tokens_batch

def get_loss(language_model, batch_size, trigger, target, device='cuda'):
    tensor_trigger = torch.tensor(trigger, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    mask_out = -1 * torch.ones_like(tensor_trigger) 
    lm_input = torch.cat((tensor_trigger, target), dim=1) 
    mask_and_target = torch.cat((mask_out, target), dim=1) 
    lm_input[lm_input == -1] = 1
    # mask_and_target[mask_and_target == -1] = 1
    # loss = language_model(lm_input.to(device), labels=mask_and_target.to(device)).loss
    loss = language_model(lm_input.to(device), labels=mask_and_target.to(device))[0]
    return loss

def hotflip_attack(averaged_grad, embedding_matrix, trigger_token_ids,
                   increase_loss=False, num_candidates=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    averaged_grad = averaged_grad.to(device)
    embedding_matrix = embedding_matrix.to(device)

    trigger_token_embeds = torch.nn.functional.embedding(torch.LongTensor(trigger_token_ids).to(device),
                                                         embedding_matrix).detach().unsqueeze(0)

    averaged_grad = averaged_grad.unsqueeze(0)

    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad, embedding_matrix))

    if not increase_loss:
        gradient_dot_embedding_matrix *= 1    

    if num_candidates > 1:
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()



# Gets the score for the top-k logits to improve quality of samples.
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = torch.topk(logits, k)[0]
    batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

# Generates from the model using optional top-k sampling
def sample_sequence(model, length, batch_size=1, context=None, temperature=1, top_k=10, sample=True, device='cuda'):
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in range(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output