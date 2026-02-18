from core.plan import ExecutionPlan


def test_plan_advance():
    plan = ExecutionPlan(["a", "b", "c"])

    assert plan.current_step() == "a"
    plan.advance()
    assert plan.current_step() == "b"
    plan.advance()
    assert plan.current_step() == "c"
    plan.advance()
    assert plan.is_finished()


def test_plan_jump():
    plan = ExecutionPlan(["a", "b", "c"])
    plan.jump_to("c")
    assert plan.current_step() == "c"
