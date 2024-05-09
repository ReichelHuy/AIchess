import random
import math
import chess

class Node:
    def __init__(self, state, player_color, parent=None, move=None):
        self.state = state
        self.player_color = player_color
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 1
        self.untried_moves = set(state.legal_moves) #list(state.legal_moves)

    def uct_select_child(self):
        """Select a child node using UCT (Upper Confidence bounds applied to Trees)."""
        log_parent_visits = math.log(self.visits)
        # Avoid division by zero by ensuring child.visits is never zero
        return max(self.children, key=lambda child: child.wins / child.visits + math.sqrt(2 * log_parent_visits / child.visits))
        
    def add_child(self, move, state):
        """Add a new child node for the move."""
        child_node = Node(state=state, player_color = self.player_color, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child_node)
        return child_node
    
    def update(self, result):
        """Update this node - one additional visit and an update to the win count."""
        self.visits += 1
        self.wins += result

def selection(node):
    """Select a node in the tree to expand."""
    while not node.state.is_game_over(claim_draw=True):
        if node.untried_moves:
            return expansion(node)
        else:
            node = node.uct_select_child()
    return node

def is_piece_threatened(move, state):
    """Determine if the piece moved is under threat before the move."""
    from_square = move.from_square
    moving_piece_color = state.color_at(from_square)
    opponent_color = not moving_piece_color

    # Check if the from_square is attacked by the opponent
    is_threatened = state.is_attacked_by(opponent_color, from_square)

    return is_threatened

def expansion(node):
    """Expand the chosen node by adding a new child."""
    """
    move = random.choice(node.untried_moves)
    new_state = node.state.copy()
    new_state.push(move)
    return node.add_child(move, new_state)  
    """
    threatened_moves = [move for move in node.untried_moves if is_piece_threatened(move, node.state)]
    safe_moves = [move for move in node.untried_moves if not is_piece_threatened(move, node.state)]
    
    # Chose from threatened pieces first
    if threatened_moves:
        move = random.choice(threatened_moves)
    else:
        move = random.choice(safe_moves) if safe_moves else None

    # Play the selected move
    new_state = node.state.copy()
    new_state.push(move)
    return node.add_child(move, new_state)

def simulation(node):
    """Simulate a random game from the given node."""
    temp_state = node.state.copy()
    move_limit = 3 #5
    moves_played = 0

    while not temp_state.is_game_over(claim_draw=True) and moves_played < move_limit:
        legal_moves = list(temp_state.legal_moves)
        # Filter moves to find captures
        capture_check_moves = [move for move in legal_moves if temp_state.is_capture(move) or temp_state.gives_check(move)]
        
        capture_check_probability = 0.75 #0.90

        if capture_check_moves and random.random() < capture_check_probability:
            selected_move = random.choice(capture_check_moves)
        else:
            selected_move = random.choice(legal_moves)
        
        temp_state.push(selected_move)
        moves_played += 1

    if temp_state.is_game_over(claim_draw=True):
        return evaluate_state(temp_state, node.player_color) 
    else:
        return evaluate_state(temp_state, node.player_color)

def backpropagation(node, result):
    """Backpropagate the result of the simulation up the tree."""
    while node is not None:
        node.update(result)
        node = node.parent

def evaluate_state(state, player_color):
    """Determine the game result from the perspective of the current player."""

    # Sum total of entire state evaluation
    score = 0

    # Check for checkmate
    if state.is_checkmate():
        if state.turn != player_color:
            score += 10000
        else:
            score += -10000
            
    # Check for draw
    elif state.is_game_over():
        score += 0 

    # Calculate value of captured or lost pieces
    last_move = state.peek()  
    if state.is_capture(last_move):
        
        captured_piece_type = state.piece_type_at(last_move.to_square)

        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        captured_piece_value = piece_values.get(captured_piece_type, 0)
        
        if state.color_at(last_move.to_square) == player_color:
            score += captured_piece_value
        else:
            score += -captured_piece_value
        
    else:
        score += 0

    # Center squares and their surrounding squares are more valuable
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    surrounding_center = [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D6, chess.E3, chess.E6, chess.F3, chess.F4, chess.F5, chess.F6]

    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

    # Penalize moving into a threatened square
    if state.is_attacked_by(not player_color, last_move.to_square):
        moved_piece_type = state.piece_type_at(last_move.to_square)
        moved_piece_value = piece_values.get(moved_piece_type, 0)
        score -= moved_piece_value * 0.5

    center_squares = set([chess.D4, chess.D5, chess.E4, chess.E5])
    surrounding_center = set([chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D6, chess.E3, chess.E6, chess.F3, chess.F4, chess.F5, chess.F6])
    attacked_center_squares = set()
    attacked_surrounding_center = set()
    
    for square in chess.SQUARES:
        piece = state.piece_at(square)
        if piece:
            piece_value = piece_values.get(piece.piece_type, 0)
            if piece.color == player_color:
                # Check if the piece is threatened, additional penalty for rooks/queen
                if state.is_attacked_by(not player_color, square):
                    score -= piece_value * 3.0 if piece_value >= 5 else 1.5
                else:
                    score += piece_value

                # Small bonus if a piece is being protected by an ally
                if state.is_attacked_by(player_color, square):
                    score += piece_value * 0.1 
                
                # Provide bonus if attacking center squares
                attacks = state.attacks(square)
                for attack_square in attacks:
                    if attack_square in center_squares:
                        attacked_center_squares.add(attack_square)
                    elif attack_square in surrounding_center:
                        attacked_surrounding_center.add(attack_square)
            else:
                # Small bonus for threatening opponent's pieces
                if state.is_attacked_by(player_color, square):
                    score += piece_value * 0.2

    score += len(attacked_center_squares) * 0.2  
    score += len(attacked_surrounding_center) * 0.1  
    
    score += evaluate_pawn_structure(state, player_color)
    score += evaluate_king_safety(state, player_color)
    score += evaluate_mobility(state, player_color)
    
    return score
    
def evaluate_mobility(state, player_color):
    mobility_score = 0
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        our_pieces = list(state.pieces(piece_type, player_color))
        for piece_square in our_pieces:
            mobility_score += len(state.attacks(piece_square))
    return mobility_score * 0.1

def evaluate_king_safety(state, player_color):
    king_safety_score = 0

    # King's position
    king_square = state.king(player_color)
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)

    # Reward maintaining castling rights
    if state.has_castling_rights(player_color):
        king_safety_score += 0.5

    # Evaluate pawn shield
    pawn_shield_bonus = 0.3
    # Check files and ranks near the king
    for file_offset in [-1, 0, 1]:  
        for rank_offset in [1, 2]: 
            if player_color == chess.WHITE:
                shield_square = chess.square(king_file + file_offset, king_rank + rank_offset)
            else:
                shield_square = chess.square(king_file + file_offset, king_rank - rank_offset)
            if shield_square in chess.SQUARES and state.piece_at(shield_square) and state.piece_at(shield_square).piece_type == chess.PAWN and state.piece_at(shield_square).color == player_color:
                king_safety_score += pawn_shield_bonus

    return king_safety_score

def evaluate_pawn_structure(state, player_color):
    pawn_structure_score = 0
    for square in chess.SQUARES:
        piece = state.piece_at(square)
        if piece and piece.piece_type == chess.PAWN and piece.color == player_color:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Search for isolated pawns
            isolated = True
            for adjacent_file in [file - 1, file + 1]:
                if 0 <= adjacent_file <= 7: 
                    for adjacent_rank in range(8):
                        if state.piece_at(chess.square(adjacent_file, adjacent_rank)) == chess.PAWN and state.piece_at(chess.square(adjacent_file, adjacent_rank)).color == player_color:
                            isolated = False
                            break
            if isolated:
                pawn_structure_score -= 0.25
            
            # Search for doubled pawns
            above_or_below = [rank + 1, rank - 1]
            for adj_rank in above_or_below:
                if 0 <= adj_rank <= 7:
                    if state.piece_at(chess.square(file, adj_rank)) == chess.PAWN and state.piece_at(chess.square(file, adj_rank)).color == player_color:
                        pawn_structure_score -= 0.25
                        break  
                
    return pawn_structure_score

def mcts(root, iterations=1000):
    for _ in range(iterations):
        leaf = selection(root)
        simulation_result = simulation(leaf)
        backpropagation(leaf, simulation_result)
    # Return the move of the best child based on the highest number of visits
    return max(root.children, key=lambda child: child.visits).move