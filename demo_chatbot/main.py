from helpers.constant import *
from helpers.utils import *

def main(model, vectorizer):
    flag = True
    while(flag):
        user_input = input()
        if user_input == 'exit':
            flag = False
        if flag:
            print(f'user : {user_input}')
            if not short_chat_rule(user_input):
                inference_result = inference(model,vectorizer,user_input)
                if is_above_threshold(inference_result):
                    response = inference_result['max_label']
                else:
                    response = 'below_proba'
            else:
                response = 'short_chat'
            print(f'bot : {response}')


if __name__ == "__main__":
    
    # load your model
    loaded_model,loaded_vectorizer = load_model()

    main(loaded_model,loaded_vectorizer)