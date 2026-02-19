
MATH_INSTRUCTION = """
You are given a mathematics problem.

Solve the problem step by step and clearly explain your reasoning.

Your response must strictly follow the format below.

R:
Provide a clear, logical, and step-by-step mathematical reasoning.
Use equations and mathematical expressions when necessary.

A:
Write only the final answer, exactly matching the required mathematical form.
Do not add explanations, units, or extra text.

Question:
""".strip()

MATH_GSM8K_INSTRUCTION = """Given a simple mathematical question, please directly provide the final answer.Question: {question};
Your response should follow the structure outlined below:
R: <Replace Here With Your Reasonings>;
A: Place your Final Answer here as a clear numeric value. Ensure there are no additional words, signs, or explanations! Enclose the numeric value in angle brackets.
An example of the desired output is:
R: First find the total number of starfish arms: 7 starfish * 5 arms/starfish = <<7*5=35>>35 arms
Then add the number of seastar arms to find the total number of arms: 35 arms + 14 arms = <<35+14=49>>49 arms
A: <49> 
"""
