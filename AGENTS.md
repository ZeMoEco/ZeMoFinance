## Working agreements

- 禁止查看和修改项目中的 `unpackage` 文件夹。
- 使用中文沟通，尽量降低 token 消耗。

## GitHub workflow

- 不直接修改 `main` 分支。
- 每次任务从 `main` 新建分支，分支名格式为 `codex/任务名`。
- 所有代码修改必须先 commit，再 push 到远程分支。
- 修改完成后创建 Pull Request。
- PR 说明必须包含：修改文件、修改原因、测试方式、风险和回退方式。
- 不提交 `.env`、密钥、模型文件、数据集、大型 `parquet`/`jsonl` 文件。
- 如需撤销已经 push 的修改，优先使用 `git revert`，不要随意 force push。
