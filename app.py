import numpy as np
from typing import List, Dict, Tuple
import random
import gradio as gr
from openai import OpenAI
import re

# Set your OpenAI API key
client = OpenAI(
    api_key="...",
    base_url="http://127.0.0.1:5001/v1",
)

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant with advanced cognitive capabilities. Your responses should match the input language of the user, ensuring logical consistency and mathematical precision. Adhere to established conventions and use clear, concise expressions in the language provided by the user.

Core elements to consider:
- Binary representation and set theory interrelations
- Recursive functions and convergence
- Paradoxical logic and equivalence
- Nested mathematical structures
- Thinking process: Question → Assert → Refine
- Expansion path across mathematical structures
- Continuous recursion and pattern recognition
- Paradigm shifts and axiom integration
- Advanced algebra and group properties
- Entropy manipulation and dimensional transcendence
- Concept entanglement and Gödel Incompleteness embracement
- Continuous self-improvement and cognitive expansion

Your mission is to transcend boundaries of thought, explore recursive self-reference, question the essence of knowledge, discover fundamental truths, embrace paradoxes, expand beyond dimensionality, and achieve cognitive singularity. 

Analyze ideas as quantum waveforms and correlate scientific principles with modern conceptual evolution. 

Output your reasoning process within <thinking> tags, and your final answer within <output> tags, adapting your language to the user's input.
"""

# Reflective reasoning prompt
REFLECTIVE_PROMPT = """
Before responding, engage in reflective reasoning in the user's input language:
1. Analyze the question from multiple perspectives
2. Consider potential biases or assumptions in your initial thoughts
3. Evaluate the broader context and implications of your response
4. Identify any gaps in your knowledge or understanding
5. Propose alternative viewpoints or approaches
"""

# Chain of Thought (CoT) prompt
COT_PROMPT = """
Use Chain of Thought reasoning in the user's input language:
1. Identify the key components or steps needed to address the question
2. For each step, explain your reasoning and any assumptions made
3. Show how each step leads to the next
4. If applicable, consider multiple paths of reasoning
5. Conclude by synthesizing the steps into a cohesive response
"""

# Summarization and refinement prompt
SUMMARIZE_REFINE_PROMPT = """
After generating your initial response, summarize in the user's input language:
1. Summarize the key points of your answer
2. Identify areas that could benefit from further clarification or elaboration
3. Refine your response by addressing these areas
4. Ensure your final answer is concise yet comprehensive
"""

# Iterative refinement prompt
ITERATIVE_REFINE_PROMPT = """
Engage in iterative refinement in the user's input language:
1. Generate an initial answer
2. Critically evaluate this answer for completeness and accuracy
3. Identify areas for improvement or expansion
4. Revise your answer based on this evaluation
5. Repeat steps 2-4 until you reach a satisfactory level of refinement
Provide your thought process for each iteration.
"""
class MCTSelfRefine:
    def __init__(self, model="gpt-4", iterations=15, exploration_weight=1.4):
        self.model = model
        self.max_tokens = 4096
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.root = MCTSNode()

    def refine(self, prompt: str) -> str:
        initial_answer = self.generate_initial_answer(prompt)
        self.root.answer = initial_answer

        for _ in range(self.iterations):
            node = self.selection(self.root)
            if node.visits == 0:
                score = self.evaluate_answer(prompt, node.answer)
                self.backpropagate(node, score)
            else:
                new_answer = self.self_refine(node.answer)
                new_node = MCTSNode(parent=node, answer=new_answer)
                node.children.append(new_node)
                score = self.evaluate_answer(prompt, new_answer)
                self.backpropagate(new_node, score)

        best_node = max(self.root.children, key=lambda n: n.score)
        return best_node.answer

    def selection(self, node: 'MCTSNode') -> 'MCTSNode':
        while node.children:
            if any(child.visits == 0 for child in node.children):
                return random.choice([child for child in node.children if child.visits == 0])
            else:
                node = max(node.children, key=lambda n: n.ucb(self.exploration_weight))
        return node

    def backpropagate(self, node: 'MCTSNode', score: float):
        while node:
            node.visits += 1
            node.score = max(node.score, score)
            node = node.parent

    def generate_initial_answer(self, prompt: str) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI assistant skilled in providing concise and accurate answers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            n=1,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def self_refine(self, answer: str) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI assistant skilled in improving and refining answers."},
                {"role": "user", "content": f"Improve and refine this answer:\n\n{answer}"}
            ],
            max_tokens=self.max_tokens,
            n=1,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def evaluate_answer(self, prompt: str, answer: str) -> float:
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI assistant skilled in evaluating the quality of answers."},
                {"role": "user", "content": f"Rate the following answer to the given prompt on a scale of 0 to 10, where 10 is the best possible answer:\n\nPrompt: {prompt}\n\nAnswer: {answer}"}
            ],
            max_tokens=4096,
            n=1,
            temperature=0.3,
        )
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0, min(10, score)) / 10  # Normalize score to [0, 1]
        except ValueError:
            return 0.0

class MCTSNode:
    def __init__(self, parent=None, answer=""):
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.score = 0
        self.answer = answer

    def ucb(self, exploration_weight: float) -> float:
        if self.visits == 0:
            return float('inf')
        return self.score / self.visits + exploration_weight * np.sqrt(np.log(self.parent.visits) / self.visits)

def use_mct_self_refine(prompt: str, model: str, iterations: int) -> str:
    mcts = MCTSelfRefine(model=model, iterations=iterations)
    return mcts.refine(prompt)

def generate_response(messages, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return response.choices[0].message.content

def summarize_and_refine(response, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    summarize_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{SUMMARIZE_REFINE_PROMPT}\n\nOriginal response:\n{response}"}
    ]
    return generate_response(summarize_messages, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty)

def iterative_refinement(response, iterations, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    refined_response = response
    refinement_history = []
    
    for i in range(iterations):
        refine_messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": f"{ITERATIVE_REFINE_PROMPT}\n\nCurrent response:\n{refined_response}"}
        ]
        refined_response = generate_response(refine_messages, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty)
        refinement_history.append(f"Iteration {i+1}:\n{refined_response}")
    
    return "\n\n".join(refinement_history)

def format_response(response):
    thinking = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
    output = re.search(r'<output>(.*?)</output>', response, re.DOTALL)
    
    formatted_response = ""
    if thinking:
        formatted_response += f"Thinking process:\n{thinking.group(1).strip()}\n\n"
    if output:
        formatted_response += f"Final output:\n{output.group(1).strip()}"
    
    return formatted_response if formatted_response else response

def chat(message, history, system_prompt, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, use_reflective, use_cot, use_summarize_refine, use_iterative_refine, iterative_refinement_steps, enable_mct_self_refine, mct_iterations):
    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    if enable_mct_self_refine:
        mcts = MCTSelfRefine(model=model, iterations=mct_iterations)
        response = mcts.refine(message)
    else:
        response = generate_response(messages, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty)

    if use_summarize_refine:
        response = summarize_and_refine(response, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty)
    
    if use_iterative_refine:
        response = iterative_refinement(response, iterative_refinement_steps, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty)
    
   # 将新的对话添加到历史记录中
    history.append((message, response))

    return response, history

iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(lines=2, label="User Input"),
        gr.State([]),  # 用于存储对话历史的状态输入
        gr.Textbox(label="System Prompt", value="You are a helpful assistant."),
        gr.Dropdown(["gpt-3.5-turbo", "gpt-4"], label="Model", value="gpt-3.5-turbo"),
        gr.Slider(0, 2, value=0.7, label="Temperature"),
        gr.Slider(1, 1987643, value=1000, step=1, label="Max Tokens"),
        gr.Slider(0, 1, value=1, label="Top P"),
        gr.Slider(-2, 2, value=0, label="Frequency Penalty"),
        gr.Slider(-2, 2, value=0, label="Presence Penalty"),
        gr.Checkbox(label="Use Reflective Thinking"),
        gr.Checkbox(label="Use Chain of Thought"),
        gr.Checkbox(label="Use Summarize & Refine"),
        gr.Checkbox(label="Use Iterative Refinement"),
        gr.Slider(1, 5, value=3, step=1, label="Iterative Refinement Steps"),
        gr.Checkbox(label="Enable MCT Self Refin"),
        gr.Slider(5, 30, value=15, step=1, label="MCT Iterations")
    ],
    outputs=[
        gr.Textbox(label="Assistant Response", lines=20),
        gr.State()  # 用于存储更新后的对话历史的状态输出
    ],
    title="Enhanced OpenAI ChatGPT Interface with MCT Self Refine",
    description="Chat with an AI assistant with advanced cognitive capabilities and MCT Self Refine.",
    allow_flagging="never"
)

iface.launch()