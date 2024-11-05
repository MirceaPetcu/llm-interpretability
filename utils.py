from transformers import AutoTokenizer
import pandas as pd


def get_word_by_index(index: int) -> list[str]:
    """
    :param index:
    :return: get the word at the index position from each sentence in the csv file
    """
    df = pd.read_csv('bold_response_LH.csv')
    words_of_interest = []
    for i, row in df.iterrows():
        words = row.iloc[1].split()
        word_of_interest = words[index] if index < len(words) else words[-1]
        words_of_interest.append(word_of_interest)
    return words_of_interest


def expand_tokens_idx(token_of_interest_idx: int, num_samples)-> list[int]:
    token_of_interest_idx = [token_of_interest_idx] * num_samples
    return token_of_interest_idx


def prepare_input(inputs: list[dict], model_id: str,
                  words_of_interest: list | str | None = None,
                  token_of_interest_idx: int | None = None) -> list[dict]:
    """
    Returns the mean embedding of the tokens corresponding to the words of interest in the input text.
    """

    if words_of_interest is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if isinstance(words_of_interest, str):
            word_tokens = tokenizer(words_of_interest, return_tensors='pt')['input_ids'][0][1:]
        new_inputs = []
        for j, inpt in enumerate(inputs):
            tokens = inpt['tokens'].to('cpu')
            if isinstance(words_of_interest, list):
                word_tokens = tokenizer(words_of_interest[j], return_tensors='pt')['input_ids'][0][1:]

            start_idx = None

            for i in range(tokens.size(1) - len(word_tokens) + 1):
                if (tokens[0, i:i + len(word_tokens)] == word_tokens).sum() == len(word_tokens):
                    start_idx = i
                    break

            if start_idx is None:
                raise ValueError(f"Word '{words_of_interest[j]}' not found in the tokenized text.")

            word_embeddings = inpt['tokens_embeddings'][start_idx:start_idx + len(word_tokens)].mean(axis=0)
            inpt['word_of_interst_embedding'] = word_embeddings
            new_inputs.append(inpt)
        return new_inputs
    elif token_of_interest_idx is not None:
        token_of_interest_idx = expand_tokens_idx(token_of_interest_idx, len(inputs))
        new_inputs = []
        for j, inpt in enumerate(inputs):
            word_embeddings = inpt['tokens_embeddings'][token_of_interest_idx[j]]
            inpt['word_of_interst_embedding'] = word_embeddings
            new_inputs.append(inpt)
        return new_inputs
    else:
        return inputs
