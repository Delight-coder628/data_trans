# 365 对 LoRA 的固定 PIE 编辑对照集

`holdout_12.json` 从 PIE-Bench `test_1536.json` 选出 12 张源图：添加、删除、替换各 4 张。全部避开此前蓝区审查的 132 个 PIE ID；筛选只查看源图、指令和 mask，未查看任何模型结果。图片不在此仓库，黄区按 `image_path` 到 PIE-Bench 原图目录查找。

为节省时间，第一轮先跑以下 6 张，每类 2 张：

| 类别 | PIE ID | 指令 |
|---|---|---|
| 添加 | `000000000138` | Add a hat to the cat |
| 添加 | `212000000005` | Add earrings to the woman |
| 删除 | `313000000002` | Remove the lit candle from the cupcake |
| 删除 | `324000000003` | Remove the small mushroom from the pine branch |
| 替换 | `111000000006` | Change the animal on the roof from a cat to a dog |
| 替换 | `122000000002` | Change the camera to a phone |

对每张源图分别运行基座、40 对 smoke LoRA（100 步）、365 对 LoRA（300 步），统一尺寸、推理步数、guidance 和随机种子。每个模型的输出单独保存为 `<sample_id>.png`。先制作并查看源图与三种输出的并排图，记录指令完成度和非编辑区变化；若差异不清，再扩到 `holdout_12.json` 的全部 12 张，必要时补跑 365 对的 100／200 步 checkpoint。

本清单中的 PIE mask 供非编辑区指标使用；其中个别添加任务的 mask 较宽，需同时做视觉判断。不要将本清单加入训练。
