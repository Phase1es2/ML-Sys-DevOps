#!/bin/bash

echo "📂 Scanning /mnt/object ..."
echo ""

du -sh /mnt/object/* 2>/dev/null | sort -h
