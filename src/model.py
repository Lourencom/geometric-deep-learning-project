import torch
import os
from utils import get_git_root, relative_to_absolute_path
from prompts import Prompts
from constants import Constants

import transformers
from transformers import AutoTokenizer


def run_model(args):
    # Unpack the current model tuple
    family, size, variant = args.current_model
    model, tokenizer = get_model_and_tokenizer(family, size, variant)

    prompts = Prompts(args.prompt_path)       
    prompt = prompts.get_prompt(
        prompt_id=args.prompt_id,
        difficulty=args.prompt_difficulty,
        category=args.prompt_category,
        n_shots=args.prompt_n_shots
    )

    prompt_text = prompt['prompt']
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        prompt_outputs = model(**inputs, output_attentions=True)

    with torch.no_grad():
        # Get the length of input tokens to separate prompt from generation
        input_length = inputs.input_ids.shape[1]
        
        # Use generate with streaming to get attention for each token
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=1,  # Disable beam search for token-by-token attention (beam search complicates attention tracking)
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,  # Get scores for each token
            output_attentions=True,  # Get attention matrices
            output_hidden_states=True,  # Get hidden states
        )
        
        # Extract attention matrices from generation
        attention_matrices = []
        if hasattr(generated, 'attentions') and generated.attentions is not None:
            # For models that return attentions directly
            attention_matrices = generated.attentions
        elif hasattr(generated, 'decoder_attentions') and generated.decoder_attentions is not None:
            # For encoder-decoder models
            attention_matrices = generated.decoder_attentions
            
        # Process attention matrices for each generated token
        token_attentions = process_token_attentions(attention_matrices)
        
        # Only decode the newly generated tokens
        answer = tokenizer.decode(generated.sequences[0][input_length:], skip_special_tokens=True)
    
    # Store both the raw generated output and processed attention matrices
    intermediate_outputs = {
        'generated': generated,
        'token_attentions': token_attentions
    }

    store_answer(args.current_model, args.output_dir, answer, prompt_text, args.prompt_id)
    
    return [prompt_outputs, intermediate_outputs], answer, prompt_text, prompt['difficulty'], prompt['category'], prompt['n_shots']

def process_token_attentions(attention_matrices):
    """
    Process the attention matrices to extract per-token attention patterns.
    
    Args:
        attention_matrices: The attention matrices from the model generation
        
    Returns:
        A list of processed attention data for each generated token
    """
    # This structure will depend on the exact format of attention matrices
    # from your specific model, but here's a general approach
    token_attentions = []
    
    if not attention_matrices:
        return token_attentions
        
    # The structure of attention matrices varies by model architecture
    # This is a general approach that may need adjustment for your specific model
    try:
        # For most models using generate with output_attentions=True
        # attention_matrices is typically a tuple of tuples
        for step_idx, step_attentions in enumerate(attention_matrices):
            # Each step corresponds to generating one new token
            # step_attentions contains attention for all layers for this step
            
            # Aggregate attention across layers (you can modify this strategy)
            # For example, you might want to look at specific layers or heads
            layer_attentions = []
            for layer_idx, layer_attention in enumerate(step_attentions):
                # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
                # We typically have batch_size=1 in generation
                
                # Average across attention heads (or select specific heads)
                # Shape becomes [seq_len, seq_len]
                avg_attention = layer_attention[0].mean(dim=0)
                layer_attentions.append(avg_attention)
            
            # Store attention for this token generation step
            token_attentions.append({
                'token_idx': step_idx,
                'layer_attentions': layer_attentions,
                # You can add more metrics here, like entropy, max attention, etc.
            })
    except Exception as e:
        # Fallback for different attention matrix structures
        print(f"Error processing attention matrices: {e}")
        # You might need to implement different processing logic
        # depending on your model's output structure
    
    return token_attentions

def store_answer(current_model, output_dir, answer, prompt_text, prompt_id):
    output_dir = relative_to_absolute_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the prompt and the answer to a text file for reference
    answer_file = os.path.join(output_dir, f"generated_answer_{current_model}_{prompt_id}.txt")
    with open(answer_file, "w") as f:
        f.write("Prompt:\n")
        f.write(prompt_text + "\n\n")
        f.write("Generated Answer:\n")
        f.write(answer)
    return answer_file

def store_attention_data(args, token_attentions, prompt_id):
    """
    Store the attention data for later analysis.
    
    Args:
        args: The command line arguments
        token_attentions: The processed attention data for each token
        prompt_id: The ID of the prompt used for generation
    
    Returns:
        The path to the saved attention data file
    """
    git_root_path = get_git_root()
    output_dir = os.path.join(git_root_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the attention data to a PyTorch file
    attention_file = os.path.join(output_dir, f"attention_data_{args.current_model}_{prompt_id}.pt")
    torch.save(token_attentions, attention_file)
    
    return attention_file

def get_model_and_tokenizer(family, size, variant):
    """
    family: str, one of ["llama", "bloom"]
    size: str, one of ["small", "large", "huge"] for llama,
          or one of ["560m", "1b1", "1b7", "3b", "7b1"] for bloom
    variant: str, either "causal" or "instruct"
    """
    if family.lower() == "llama":
        # existing Llama code, e.g.
        if size == "small":
            if variant == "causal":
                model_id = Constants.LLAMA_SMALL_CAUSAL
            elif variant == "instruct":
                model_id = Constants.LLAMA_SMALL_INSTRUCT
            else:
                raise ValueError("Invalid variant for llama")
        elif size == "large":
            if variant == "causal":
                model_id = Constants.LLAMA_LARGE_CAUSAL
            elif variant == "instruct":
                model_id = Constants.LLAMA_LARGE_INSTRUCT
            else:
                raise ValueError("Invalid variant for llama")
        elif size == "huge":
            if variant == "instruct":
                model_id = Constants.LLAMA_HUGE_INSTRUCT
            else:
                raise ValueError("No causal huge variant for llama")
        else:
            raise ValueError("Invalid llama size")
    
    elif family.lower() == "bloom":
        # For BLOOM, use the corresponding string IDs
        if size == "560m":
            if variant == "causal":
                model_id = Constants.BLOOM_560M_CAUSAL
            elif variant == "instruct":
                model_id = Constants.BLOOM_560M_INSTRUCT
            else:
                raise ValueError("Invalid variant for bloom")
        elif size == "1b1":
            if variant == "causal":
                model_id = Constants.BLOOM_1B1_CAUSAL
            elif variant == "instruct":
                model_id = Constants.BLOOM_1B1_INSTRUCT
            else:
                raise ValueError("Invalid variant for bloom")
        elif size == "1b7":
            if variant == "causal":
                model_id = Constants.BLOOM_1B7_CAUSAL
            elif variant == "instruct":
                model_id = Constants.BLOOM_1B7_INSTRUCT
            else:
                raise ValueError("Invalid variant for bloom")
        elif size == "3b":
            if variant == "causal":
                model_id = Constants.BLOOM_3B_CAUSAL
            elif variant == "instruct":
                model_id = Constants.BLOOM_3B_INSTRUCT
            else:
                raise ValueError("Invalid variant for bloom")
        elif size == "7b1":
            if variant == "causal":
                model_id = Constants.BLOOM_7B1_CAUSAL
            elif variant == "instruct":
                model_id = Constants.BLOOM_7B1_INSTRUCT
            else:
                raise ValueError("Invalid variant for bloom")
        else:
            raise ValueError("Invalid bloom size")
    else:
        raise ValueError("Unsupported model family")
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def sample_next_token(current_ids, next_token_logits, temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.0):
    
    # Apply temperature
    if temperature > 0:
        next_token_logits = next_token_logits / temperature
    
    # Apply repetition penalty
    if repetition_penalty != 1.0:
        for token_id in current_ids[0]:
            next_token_logits[0, token_id] /= repetition_penalty
    
    # Apply top-k filtering
    if top_k > 0:
        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
        next_token_logits[indices_to_remove] = -float('Inf')
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[0, indices_to_remove] = -float('Inf')
    
    # Sample from the filtered distribution
    probs = torch.softmax(next_token_logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    return next_token_id


def get_token_by_token_attention(model, tokenizer, prompt_text, max_new_tokens=512):
    """
    Captures the full attention matrices for each generated token across all layers.
    
    This function performs token-by-token generation and captures the attention matrices
    at each step, preserving the full attention state (all heads, all layers) for each
    token generation step.
    
    Args:
        model: The HuggingFace model to use for generation
        tokenizer: The tokenizer corresponding to the model
        prompt_text: The input prompt text
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        A dictionary containing:
        - 'prompt_tokens': The tokenized prompt
        - 'generated_tokens': The generated tokens
        - 'full_text': The complete text (prompt + generation)
        - 'attention_matrices': A list of attention matrices for each generation step
          Each element is a list of layer attentions, where each layer attention is
          a tensor of shape [batch_size, num_heads, curr_seq_len, curr_seq_len]
        - 'token_ids': The token IDs for each step of generation
        - 'token_texts': The decoded text for each generated token
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    input_length = input_ids.shape[1]
    
    # Store the results
    attention_matrices = []
    generated_token_ids = []
    token_texts = []
    
    # Start with the prompt tokens
    current_ids = input_ids.clone()
    
    # Generate tokens one by one
    with torch.no_grad():
        # First get prompt attention
        prompt_outputs = model(current_ids, output_attentions=True, return_dict=True)
        prompt_attentions = prompt_outputs.attentions
        
        # Then generate tokens
        for _ in range(max_new_tokens):
            # Forward pass with attention outputs
            outputs = model(current_ids, output_attentions=True, return_dict=True)
            
            # Get the next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = sample_next_token(current_ids, next_token_logits)
            
            # Store the attention matrices for this step
            # This captures attention for all layers and all heads
            # Each attention matrix is of shape [batch_size, num_heads, curr_seq_len, curr_seq_len]
            step_attentions = outputs.attentions
            attention_matrices.append(step_attentions)
            
            # Store the generated token
            generated_token_ids.append(next_token_id.item())
            token_text = tokenizer.decode(next_token_id.item())
            token_texts.append(token_text)
            
            # Check if we've hit the end of sequence token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
            # Append the new token to the sequence
            current_ids = torch.cat([current_ids, next_token_id], dim=1)
    
    # Decode the full generated text
    full_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(current_ids[0][input_length:], skip_special_tokens=True)
    
    return {
        'prompt_tokens': input_ids,
        'prompt_attentions': prompt_attentions,
        'generated_tokens': generated_token_ids,
        'full_text': full_text,
        'generated_text': generated_text,
        'attention_matrices': attention_matrices,
        'token_texts': token_texts,
        'prompt_text': prompt_text,
    }


def get_tokenwise_attns(current_model, prompt="Explain the concept of attention in transformer models."):
    """
    Example function demonstrating how to use get_token_by_token_attention.
    
    Args:
        model_family: The model family to use (e.g., "llama", "bloom")
        model_size: The model size to use (e.g., "small", "large")
        model_variant: The model variant to use (e.g., "causal", "instruct")
        prompt: The prompt text to use
        
    Returns:
        The attention data from get_token_by_token_attention
    """
    # Get the model and tokenizer
    model_family, model_size, model_variant = current_model
    model, tokenizer = get_model_and_tokenizer(model_family, model_size, model_variant)
    
    # Get token-by-token attention

    attention_data = get_token_by_token_attention(model, tokenizer, prompt, max_new_tokens=50)        
    
    return attention_data

def save_attention_data(attention_data, output_path):
    """
    Saves the attention data to disk for later analysis.
    
    Args:
        attention_data: The output from get_token_by_token_attention
        output_path: Path to save the data
        
    Returns:
        The path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert attention matrices to CPU before saving
    cpu_attention_data = attention_data.copy()
    
    # Process attention matrices
    cpu_attention_matrices = []
    for step_attentions in attention_data['attention_matrices']:
        cpu_step_attentions = []
        for layer_attention in step_attentions:
            # Move to CPU
            cpu_layer_attention = layer_attention.cpu()
            cpu_step_attentions.append(cpu_layer_attention)
        cpu_attention_matrices.append(cpu_step_attentions)
    
    cpu_attention_data['attention_matrices'] = cpu_attention_matrices
    
    # Move prompt tokens to CPU
    cpu_attention_data['prompt_tokens'] = attention_data['prompt_tokens'].cpu()
    
    # Save to disk
    torch.save(cpu_attention_data, output_path)
    
    return output_path

def load_attention_data(input_path, device=None):
    """
    Loads the attention data from disk.
    
    Args:
        input_path: Path to the saved attention data
        device: Device to load the tensors to (None for CPU)
        
    Returns:
        The loaded attention data
    """
    # Load from disk
    attention_data = torch.load(input_path)
    
    # Move to specified device if needed
    if device is not None:
        # Process attention matrices
        device_attention_matrices = []
        for step_attentions in attention_data['attention_matrices']:
            device_step_attentions = []
            for layer_attention in step_attentions:
                # Move to device
                device_layer_attention = layer_attention.to(device)
                device_step_attentions.append(device_layer_attention)
            device_attention_matrices.append(device_step_attentions)
        
        attention_data['attention_matrices'] = device_attention_matrices
        
        # Move prompt tokens to device
        attention_data['prompt_tokens'] = attention_data['prompt_tokens'].to(device)
    
    return attention_data

def batch_analyze_prompts(model, tokenizer, prompts, output_dir=None, max_new_tokens=100):
    """
    Analyzes multiple prompts and collects their attention patterns.
    
    Args:
        model: The HuggingFace model to use
        tokenizer: The tokenizer corresponding to the model
        prompts: List of prompt texts to analyze
        output_dir: Directory to save the results (None to skip saving)
        max_new_tokens: Maximum number of tokens to generate for each prompt
        
    Returns:
        A dictionary mapping prompt indices to attention data
    """
    results = {}
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        
        # Get token-by-token attention
        attention_data = get_token_by_token_attention(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        
        # Store the results
        results[i] = {
            'prompt': prompt,
            'attention_data': attention_data
        }
        
        # Save to disk if output directory is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"prompt_{i}_attention.pt")
            save_attention_data(attention_data, output_path)
            results[i]['saved_path'] = output_path
    
    return results
