from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

def mul_three(x: int) -> int:
    return x * 3

runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)
runnable_3 = RunnableLambda(mul_three)

sequence = runnable_1 | {  # this dict is coerced to a RunnableParallel
    "mul_two": runnable_2,
    "mul_three": runnable_3,
}
# Or equivalently:
# sequence = runnable_1 | RunnableParallel(
#     {"mul_two": runnable_2, "mul_three": runnable_3}
# )
# Also equivalently:
# sequence = runnable_1 | RunnableParallel(
#     mul_two=runnable_2,
#     mul_three=runnable_3,
# )

sequence.invoke(1)
await sequence.ainvoke(1)

sequence.batch([1, 2, 3])
await sequence.abatch([1, 2, 3])