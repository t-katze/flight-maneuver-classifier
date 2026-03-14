# WindowsのZone.Identifierを一括削除するコマンド
# まず確認（どれだけあるか）
find . -type f -name '*Zone.Identifier*' -print | head
# 削除
find . -type f -name '*Zone.Identifier*' -delete
