from verl.utils.reward_score.gsm8k_v1 import compute_score

ground_truth = "72"

cases = [
    {
        "name": "格式正确，答案正确",
        "solution": "Natalia sold 24 clips in May. So total is 48 + 24 = 72.\n#### 72",
        "expected": 1.1,
    },
    {
        "name": "格式正确，答案错误",
        "solution": "Natalia sold 24 clips in May. So total is 48 + 24 = 96.\n#### 96",
        "expected": 0.1,
    },
    {
        "name": "格式错误，但答案正确",
        "solution": "Natalia sold 24 clips in May. So total is 48 + 24 = 72. The answer is 72.",
        "expected": 1.0,
    },
    {
        "name": "格式错误，答案错误",
        "solution": "Natalia sold 24 clips in May. So total is 96.",
        "expected": 0.0,
    },
    {
        "name": "没有数字",
        "solution": "I do not know the answer.",
        "expected": 0.0,
    },
]

for case in cases:
    score = compute_score(
        solution_str=case["solution"],
        ground_truth=ground_truth,
        format_score=0.1,
        score=1.0,
    )
    print(f'{case["name"]}: score={score}, expected={case["expected"]}')
    assert abs(score - case["expected"]) < 1e-6, case

print("All reward tests passed.")
