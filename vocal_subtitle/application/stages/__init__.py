"""Pipeline 生命周期阶段对象(2026-09-15 重构计划 Task 3)。

阶段对象把 ``run_lifecycle.run()`` 的线性阶段编排显式化:
每个阶段接收 pipeline(组合现有 Mixin 方法)与 ``RunContext``,
执行结果写入 ``context.add_diagnostic`` 与 ``context.state``。
"""
