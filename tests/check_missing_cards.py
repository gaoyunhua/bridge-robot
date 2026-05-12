#!/usr/bin/env python3
"""检查52张牌缺哪两张"""
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from endplay.types import Card, Denom, Rank


def parse_card(s):
    """Parse card string like 'JC', '2D', 'AH'"""
    rank_map = {'2': Rank.R2, '3': Rank.R3, '4': Rank.R4, '5': Rank.R5, '6': Rank.R6, '7': Rank.R7, '8': Rank.R8, '9': Rank.R9, 'T': Rank.RT, 'J': Rank.RJ, 'Q': Rank.RQ, 'K': Rank.RK, 'A': Rank.RA}
    suit_map = {'C': Denom.clubs, 'D': Denom.diamonds, 'H': Denom.hearts, 'S': Denom.spades}

    rank = rank_map[s[0]]
    suit = suit_map[s[1]]
    return Card(suit, rank)


def main():
    # Play History from Terminal 1061-1062
    play_history_str = "JC → 9C → KC → AC → 2H → AH → 6H → QH → AD → JD → TD → KD → 9S → QS → AS → JS → 5C → KH → QC → 7C → 6S → 5S → KS → 3S → 3C → 8D → TC → 6C → 9H → TH → 5H → JH → 8H → 8C → 4C → 7H → 4H → TS → QD → 5D → 9D → 6D → 7D → 4D → 3H → 8S → 3D → 2D → 4S → 7S"

    # Parse cards
    cards_str = play_history_str.split(' → ')
    play_history_cards = set()
    for c in cards_str:
        c = c.strip()
        if c:
            card = parse_card(c)
            play_history_cards.add(card)

    print(f"Play History 有 {len(play_history_cards)} 张不同的牌")

    # 所有52张牌
    all_cards = set()
    for suit in [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades]:
        for rank in [Rank.R2, Rank.R3, Rank.R4, Rank.R5, Rank.R6, Rank.R7, Rank.R8, Rank.R9, Rank.RT, Rank.RJ, Rank.RQ, Rank.RK, Rank.RA]:
            all_cards.add(Card(suit, rank))

    print(f"总共 {len(all_cards)} 张牌")

    # 找出缺少的牌
    missing = all_cards - play_history_cards
    print(f"\n缺少 {len(missing)} 张牌：")
    for c in sorted(missing, key=lambda x: (x.suit.value, x.rank.value)):
        print(f"  {c}")


if __name__ == "__main__":
    main()