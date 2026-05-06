@echo off
ssh sshuser@192.168.0.160 "cd D:\awbw && powershell -Command \"Get-Content logs\\games_log.jsonl -First 100 2>&1\""
