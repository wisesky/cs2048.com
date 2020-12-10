---
title: LeetCode 115. Distinct Subsequences
toc: true
mathjax: true
top: false
cover: true
date: 2020-12-09 17:52:16
updated:
categories:
tags:
---

> Given two strings `s` and `t`, return *the number of distinct subsequences of `s` which equals `t`*.
>
> A string's **subsequence** is a new string formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., `"ACE"` is a subsequence of `"ABCDE"` while `"AEC"` is not).
>
> It's guaranteed the answer fits on a 32-bit signed integer.
>
>  
>
> **Example 1:**
>
> ```
> Input: s = "rabbbit", t = "rabbit"
> Output: 3
> Explanation:
> As shown below, there are 3 ways you can generate "rabbit" from S.
> rabbbit
> rabbbit
> rabbbit
> ```
>
> **Example 2:**
>
> ```
> Input: s = "babgbag", t = "bag"
> Output: 5
> Explanation:
> As shown below, there are 5 ways you can generate "bag" from S.
> babgbag
> babgbag
> babgbag
> babgbag
> babgbag
> ```
>
>  
>
> **Constraints:**
>
> - `0 <= s.length, t.length <= 1000`
> - `s` and `t` consist of English letters.



　　或许是由于之前做过这个题目的原因，所以面对这道Hard题目，心态比较冷静；再加上100题前后的几乎都是二叉树类别的题目，自己面对链表和树类别的题目基本没碰到困难，所以DFS用的手软，到这里有点停不下来的感觉。所以此时第一想法就想到用DFS，虽然预感对于字符串匹配类别的题目，DFS肯定会出现性能问题，但是作为验证算法思想的先导思路，还是打算先实现出来再看，毕竟用DFS写代码，代码真的很短，AC的时候更是有种 吊炸天的 爽快感；结果还是差一点AC；其次，可能是之前做过的原因，虽然第一时间没写出dp递推公式，但是写出了递推矩阵，演算了一番，没想到居然是对的，很容易的转换成了算法代码，结果干净利落的AC了，甚至比之前的AC的时候还要快一倍以上。虽然这不是自己第一次独立解决Hard题目，但是这题没有借鉴之前的代码，而且从构思算法道实践成功基本没遇到什么障碍，还是值得自己兴奋一阵子了。

### 解法一：DFS

　　