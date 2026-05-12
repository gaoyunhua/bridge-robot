#!/usr/bin/env python3
"""
Bridge Game UI Server using Endplay EXCLUSIVELY for all Bridge logic!
"""
import sys
import os
import json
from flask import Flask, render_template, jsonify, request

from endplay.types import Bid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bridge_board import ExtendedBoard


app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(__file__), 'static'),
            static_url_path='/static')

# Global game state (uses endplay ExtendedBoard as primary state store)
game_board = None


@app.route('/')
def index():
    return render_template('endplay.html')


@app.route('/api/start', methods=['POST'])
def start_game():
    global game_board
    
    try:
        req = request.get_json() or {}
        mode = req.get('mode', '4ai')
        
        game_board = ExtendedBoard.new_game(mode=mode)
        
        return jsonify({
            'success': True,
            'state': game_board.get_game_state()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/state')
def get_state_route():
    return jsonify({
        'success': True,
        'state': game_board.get_game_state() if game_board else {}
    })


@app.route('/api/bid', methods=['POST'])
def make_bid():
    global game_board
    
    if not game_board:
        return jsonify({
            'success': False,
            'error': 'Game not initialized'
        }), 400
        
    if not game_board.is_bidding():
        return jsonify({
            'success': False,
            'error': 'Not in bidding phase'
        }), 400
        
    try:
        req = request.get_json() or {}
        bid_str = req.get('bid')
        
        if not bid_str:
            return jsonify({
                'success': False,
                'error': 'No bid specified'
            }),400
            
        new_bid = Bid(bid_str)
        game_board.auction.append(new_bid)
        
        if not game_board.is_bidding():
            print(f'Bidding complete! Contract: {game_board.get_contract()}')
        
        return jsonify({
            'success': True,
            'state': game_board.get_game_state()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }),500


@app.route('/api/ai_bid')
def ai_bid():
    """Smart AI to make bids using DDS analysis"""
    global game_board
    
    if not game_board or not game_board.is_bidding():
        return jsonify({
            'success': False,
            'error': 'Not in bidding phase'
        }), 400
        
    try:
        selected, _, source = game_board.make_smart_bid()
        if selected is None:
            return jsonify({
                'success': False,
                'error': 'Bidding is complete'
            }), 400
        
        new_bid = Bid(selected)
        game_board.auction.append(new_bid)
        game_board.bid_sources.append(source)
        
        if not game_board.is_bidding():
            print(f'Bidding complete! Contract: {game_board.get_contract()}')
        
        return jsonify({
            'success': True,
            'state': game_board.get_game_state()
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }),500


@app.route('/api/validate_bid', methods=['POST'])
def validate_bid():
    try:
        req = request.get_json() or {}
        bid_str = req.get('bid')
        valid = game_board.is_legal_bid(bid_str) if game_board else False
        return jsonify({
            'success': True,
            'valid': valid,
            'bid': bid_str
        })
    except Exception:
        return jsonify({
            'success': True,
            'valid': False,
            'bid': bid_str if 'bid_str' in locals() else ''
        })


@app.route('/api/play_card', methods=['POST'])
def play_card():
    try:
        req = request.get_json() or {}
        card_str = req.get('card')
        
        if not game_board or not game_board.is_playing():
            return jsonify({
                'success': False,
                'error': 'Not in playing phase'
            }), 400
        
        current_player = game_board.get_current_player_idx()
        
        success = game_board.play_card(current_player, card_str)
        
        return jsonify({
            'success': success,
            'state': game_board.get_game_state()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ai_play')
def ai_play():
    try:
        if not game_board or not game_board.is_playing():
            return jsonify({
                'success': False,
                'error': 'Not in playing phase'
            }), 400
        
        current_player = game_board.get_current_player_idx()
        legal_plays = game_board.get_legal_plays(current_player)
        
        if not legal_plays:
            return jsonify({
                'success': False,
                'error': 'No legal plays'
            }), 400
        
        card = legal_plays[0]
        game_board.play_card(current_player, str(card))
        
        return jsonify({
            'success': True,
            'state': game_board.get_game_state()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/legal_plays')
def legal_plays():
    try:
        if not game_board or not game_board.is_playing():
            return jsonify({
                'success': False,
                'error': 'Not in playing phase'
            }), 400
        
        current_player = game_board.get_current_player_idx()
        legal_cards = game_board.get_legal_plays(current_player)
        legal_strings = [str(c) for c in legal_cards]
        
        return jsonify({
            'success': True,
            'cards': legal_strings
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print('=' * 80)
    print('Bridge Game UI Server using ENDPLAY EXCLUSIVELY!')
    print('=' * 80)
    print('All bridge logic handled by endplay (no custom code!)')
    print('Features:')
    print('  - Bidding using endplay.Board.auction')
    print('  - Play using endplay.Deal')
    print('  - Vulnerability, dealer, contracts handled')
    print('  - Slice-stacked cards display')
    print('  - Smart AI bidding using DDS analysis')
    print('  - DDS results cached for performance')
    print('=' * 80)
    print('Starting server at http://localhost:5015')
    print('=' * 80)
    app.run(debug=True, host='0.0.0.0', port=5015, use_reloader=False)
