# [WAY博](https://way-adventurer.github.io/)

## 指令

### 1.本地预览

```
bundle install

bundle exec jekyll serve
```

### 2. 提交文件

```bash
# 添加所有文件到暂存区
git add .

# 提交更改
git commit -m "初始化博客"

# 设置默认分支为 main（GitHub 现在默认使用 main）
git branch -M main

# 推送到远程仓库
git push -u origin main
```

### 3.查看更改文件

```
git status
```

### 4.初始化本地仓库

```
# 删除原有的 Git 配置
Remove-Item -Path .git -Recurse -Force

# 初始化新的 Git 仓库
git init

# 添加远程仓库
git remote add origin https://github.com/way-adventurer/way-adventurer.github.io
```

### 5.设置代理

```
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

































