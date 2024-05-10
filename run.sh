

# python task/index_op.py --model openai-community/gpt2 --model_alias gpt2.pkl
# python task/index_op.py --model microsoft/phi-2 --model_alias phi-2.pkl
python index_op.py  --psize 4 --model openai-community/gpt2-xl --model_alias gpt2-xl.pkl
python index_op.py  --psize 4 --model EleutherAI/gpt-j-6b --model_alias gpt-j-6b.pkl --revision float16
python index_op.py  --psize 4 --model mistralai/Mistral-7B-v0.1 --model_alias Mistral-7B-v0.1.pkl
python index_op.py  --psize 4 --model meta-llama/Llama-2-7b-hf --model_alias Llama-2-7b-hf.pkl
python index_op.py  --psize 4 --model meta-llama/Llama-2-13b-hf --model_alias Llama-2-13b-hf.pkl
python index_op.py  --psize 4 --model stabilityai/stablelm-2-1_6b --model_alias stablelm-2-1_6b.pkl
python index_op.py  --psize 4 --model microsoft/phi-1_5 --model_alias phi-1_5.pkl
python index_op.py  --psize 4 --model HuggingFaceH4/zephyr-7b-beta --model_alias zephyr-7b-beta.pkl

python index_op.py  --psize 4 --no-task-description --model openai-community/gpt2-xl --model_alias gpt2-xl.pkl
python index_op.py  --psize 4 --no-task-description --model EleutherAI/gpt-j-6b --model_alias gpt-j-6b.pkl --revision float16
python index_op.py  --psize 4 --no-task-description --model mistralai/Mistral-7B-v0.1 --model_alias Mistral-7B-v0.1.pkl
python index_op.py  --psize 4 --no-task-description --model meta-llama/Llama-2-7b-hf --model_alias Llama-2-7b-hf.pkl
python index_op.py  --psize 4 --no-task-description --model meta-llama/Llama-2-13b-hf --model_alias Llama-2-13b-hf.pkl
python index_op.py  --psize 4 --no-task-description --model stabilityai/stablelm-2-1_6b --model_alias stablelm-2-1_6b.pkl
python index_op.py  --psize 4 --no-task-description --model microsoft/phi-1_5 --model_alias phi-1_5.pkl
python index_op.py  --psize 4 --no-task-description --model HuggingFaceH4/zephyr-7b-beta --model_alias zephyr-7b-beta.pkl

# python token_manipulation.py --model microsoft/phi-2 --model_alias phi-2.pkl
# python token_manipulation.py --model openai-community/gpt2 --model_alias gpt2.pkl
# python token_manipulation.py --model openai-community/gpt2-xl --model_alias gpt2-xl.pkl
# python token_manipulation.py --model EleutherAI/gpt-j-6b --model_alias gpt-j-6b.pkl --revision float16
# python token_manipulation.py --model mistralai/Mistral-7B-v0.1 --model_alias Mistral-7B-v0.1.pkl
# python token_manipulation.py --model meta-llama/Llama-2-7b-hf --model_alias Llama-2-7b-hf.pkl
# python token_manipulation.py --model meta-llama/Llama-2-13b-hf --model_alias Llama-2-13b-hf.pkl
# python token_manipulation.py --model stabilityai/stablelm-2-1_6b --model_alias stablelm-2-1_6b.pkl
# python token_manipulation.py --model microsoft/phi-1_5 --model_alias phi-1_5.pkl
# python token_manipulation.py --model HuggingFaceH4/zephyr-7b-beta --model_alias zephyr-7b-beta.pkl


