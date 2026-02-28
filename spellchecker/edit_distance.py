"""
edit_distance.py
----------------
Implements the Levenshtein (edit) distance algorithm:
    - Insertion  : cost 1
    - Deletion   : cost 1
    - Substitution (same char) : cost 0
    - Substitution (diff char) : cost 2

Time  complexity : O(m * n)  where m = len(s1), n = len(s2)
Space complexity : O(m * n)  (can be reduced to O(min(m,n)))
"""

def levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(m + 1):
        dp[i][0] = i

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            ch1, ch2 = s1[i - 1], s2[j - 1]
            sub_cost = 0 if ch1 == ch2 else 2

            dp[i][j] = min(
                dp[i - 1][j] + 1,          
                dp[i][j - 1] + 1,          
                dp[i - 1][j - 1] + sub_cost  
            )
    return dp[m][n]


# def edit_distance_matrix(s1: str, s2: str) -> list:
#     m, n = len(s1), len(s2)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#     for j in range(n + 1):
#         dp[0][j] = j
#     for i in range(m + 1):
#         dp[i][0] = i
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             sub_cost = 0 if s1[i - 1] == s2[j - 1] else 2
#             dp[i][j] = min(
#                 dp[i - 1][j] + 1,
#                 dp[i][j - 1] + 1,
#                 dp[i - 1][j - 1] + sub_cost
#             )
#     return dp