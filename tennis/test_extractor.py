import sys
sys.path.insert(0, r'c:\Users\trumen001\OneDrive - University of Surrey\Desktop\backtest_match\tennis')
from betfair_odds_extractor import extract_combined_total_file, scan_event_file

# Test on the 2018 event we found
event_dir = r'c:\Users\trumen001\OneDrive - University of Surrey\Desktop\backtest_match\tennis\odd_data\2018\Apr\1\28660165'
event_bz2 = event_dir + r'\28660165.bz2'

player_a, player_b, total_markets = scan_event_file(event_bz2)
print(f'Players: {player_a} vs {player_b}')
print(f'Total markets: {total_markets}')

for mid, mtype in total_markets:
    fp = event_dir + f'\\{mid}.bz2'
    result = extract_combined_total_file(fp)
    print(f'{mtype} {mid}: {result}')
