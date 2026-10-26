# T3 Q-learning 复现代码

主程序：`t3_qlearning.py`

```bash
python t3_qlearning.py --experiment all --workers 8
python t3_qlearning.py --experiment quick --workers 4
python t3_qlearning.py --experiment all --replot
```

输出：
- `figures/`：论文图示的缩减规模复现；
- `results/*.csv`：每次会话结果与汇总表；
- `results/summary.json`：主要统计量；
- `results/all_results.pkl`：含脉冲响应和策略矩阵的完整缓存。

完整运行时间取决于 CPU，约 10–20 分钟。
