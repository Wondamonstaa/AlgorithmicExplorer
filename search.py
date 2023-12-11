#Name: Kiryl Baravikou
#Course: CS 411, Fall 2023
#Homework: 5
#Professor: Piotr Gmytrasiewicz


from time import time #Used for time recording
from psutil import Process #Allows to retreive system memory usage stats
from collections import deque
import psutil #Allows to retreive system memory usage stats
from bisect import insort
import heapq


#I decided to introduce the following mixin object to incorporate the memoization technique, which will reduce the 
#time complexity of BFS algorithm, and increase the efficiency of the code.
class MemoMix:
    
    #Constructor
    def __init__(self):
        self.memo = {}

class Node:
    
    #Constructor
    def __init__(self, moves, state, action, depth, gScore, heuristic):
        
        #Current move of the tile => stored in moves after appending
        self.action = action
        
        #Current depth of the selected node
        self.depth = depth
        
        #For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
        self.gScore = gScore
        self.heuristic = heuristic
        self.moves = moves
        
        #Current state of the tile
        self.state = state
        
        
    #The following function serves to initialize the BFS algorithm on the provided list of values
        def run_bfs(initial_list):
            
            #According to the algorithm in the book, we initialize an empty set first
            visited = set()
            
            #Then we initialize the queue which is served to store a list containing a single tuple, encapsulated in a deque.
            #Why did I choose deque: it allows for efficient adding + removing of elements from both ends of the queue.
            #Deque will be used for storing states of a 15-puzzle game.
            queue = deque([(initial_list, [])])
            
            #Initially zero. Increment later.
            nodes_expanded = 0
            
            #Record the time when the algo starts running. Later used for finding the time taken.
            start_time = time.time()
            
            #This lambda f() takes a a list and computes a hash value for it
            hash_table = lambda board: hash(tuple(tuple(row) for row in board))

            while queue is not []:
                
                current_board, moves = queue.popleft()
                nodes_expanded += 1
        
                cur_state = hash_table(current_board)
        
                #Need to check if the current state = the goal state using goal_test()
                if self.goal_test([str(cur_tile) for _ in current_board for cur_tile in _]):
                    
                    #Here I compute the stats required for the output
                    end_time = time.time()
                    diff_time = end_time - start_time
                    VM = (psutil.virtual_memory().used / pow(1024, 2))
                    
                    return moves, nodes_expanded, diff_time, VM
        
                #Update the visited array to proceed next
                visited.add(cur_state)
        
                for adjNode, nextMove in self.getAdjNode(current_board):
                    neighbor_hash = hash_table(adjNode)
                    if neighbor_hash not in visited:
                        queue.append((adjNode, moves + [nextMove]))
        
            return None


        #Used to determine a cycle in the configuration sent
        def is_cycle(node, visited_states):
            if tuple(tuple(row) for row in node) in visited_states:
                return True
            return False

        #To keep track of what nodes we have visited so far
        visited_states = set()

        #Iterative DFS algorithm based on the providede pseudocode
        def idfs(initial_list):
            
            #Initial depth is zero
            depth = 0
            
            #9999 represents infinity
            for depth in range(0, 9999):
                
                result = ldfs(initial_list, depth)
                
                #Check if there is no cutoff. If not, we return a 4-pack of values
                if(result != 'cutoff'):
                    return result
                
            
        #The DFS algorithm with the limited depth
        def ldfs(initial_list, depth):
            
            #According to the algorithm in the book, we initialize an empty set first
            visited = set()
            
            #Initialize the frontier which must represent a stack => used deque() structure
            frontier = deque([(initial_list, [])])
            
            #Represent incorrect result
            result = -1
            
            #Initially zero. Increment later.
            nodes_expanded = 0
            
            #Used to count the current depth while we expanding the path
            depth_count = 0
            
            #Record the time when the algo starts running. Later used for finding the time taken.
            start_time = time.time()
            
            #This lambda f() takes a a list and computes a hash value for it
            hash_table = lambda board: hash(tuple(tuple(row) for row in board))
            
            #Iterate until we have elements on the stack
            while frontier is not []:
                
                current_board, moves = frontier.popleft()
                nodes_expanded += 1
                depth_count = depth_count + 1
        
                cur_state = hash_table(current_board)
                
                #Need to check if the current state = the goal state using goal_test()
                if self.goal_test([str(cur_tile) for _ in current_board for cur_tile in _]):
                    
                    #Here I compute the stats required for the output
                    end_time = time.time()
                    diff_time = end_time - start_time
                    VM = (psutil.virtual_memory().used / pow(1024, 2))
                    
                    return moves, nodes_expanded, diff_time, VM
                
                #If there is a potential solution on the deeper path that we cannot reach
                if depth_count > depth:
                    return 'cutoff'
                
                #Update the visited array to proceed next
                visited.add(cur_state)
        
                for adjNode, nextMove in self.getAdjNode(current_board):
                    
                    neighbor_hash = hash_table(adjNode)
                    
                    #Check if not visited yet
                    if neighbor_hash not in visited:
                        
                        #Check if there is no cycles, and if not, append the node and proceed
                        if is_cycle(adjNode, visited) == 0:
                            frontier.append((adjNode, moves + [nextMove]))
        
            return result
        
    
    #Helper function to compare two nodes for inequality
    def __lt__(self, that):
        
        lhs = self.gScore + self.heuristic
        rhs = that.gScore + that.heuristic
        
        return rhs > lhs
    
       
#The following class is used to store all the helper functions that will be implemented later. 
#Mixed in with MemoMix for memoization purposes.
class Storage(MemoMix):
    
    #Constructor
    def __init__(self, initial_state = None):
        
        #self.initial_state = initial_state
        
        #Call to initialize the memo attribute of the mixin class
        super().__init__()
        

    #Converts KB to bytes
    def converter(self, kilobytes):
        bytes = kilobytes * 1024
        return bytes
    
    
#The main class to run the search algorithms
class Search(Storage):
    
    #Constructor
    def __init__(self, size=4, initial_state=['1 0 2 4 5 7 3 8 9 6 11 12 13 10 14 15'], time_limit=60, heuristic="manhattan"):
        
        self.heuristic = heuristic
        self.size = size
        self.initial_state = initial_state
        self.time_limit = time_limit
        self.solution = list(range(1, 16)) + [0]
        
        #Call to initialize the memo attribute of the mixin class
        #super().__init__()
        
    
    #Driver of the program
    @classmethod
    def solve(cls, initial_state, heuristic="manhattan"):
        
        try:
            
            #Split the input string into individual values and remove extra spaces
            initial_values = initial_state.split()
            initial_list = [int(value) for value in initial_values]
    
            #Check if the input list has the correct length
            if len(initial_list) != 16:
                raise ValueError("Board configuration must contain exactly 16 tiles.")
    
            total = sum(initial_list)
            
            if total != 120:
                raise ValueError("The sum of board values must be 120.")
    
            #Agent object of the class 
            agent = cls(4, initial_list, 60, heuristic)
            
            #Stores the result of executing A* inside the var
            solution_moves = agent.A_Star()
            
            #If the var is not empty, then we pass its contents to the eval function
            if solution_moves is not None:
                return agent.process_solution_moves(solution_moves)
            else:
                print("No solution found.")
         
    
        #If the input was not valid => throw an exception
        except Exception as e:
           
           print(f"Error: {e}")
           print("Please, provide a valid input!")
           return None.__class__()
       
        except ValueError as v:
           print(f"Error: {v}")
           print("Please, provide a valid input without repetitions!")
           return None.__class__()

    
    
    #A* algorithm based on the providede pseudocode
    def A_Star(self):
        
            #Initially zero. Increment later.
            nodes_expanded = 0
            
            #Record the time when the algo starts running. Later used for finding the time taken.
            start_time = time()
            
            #This lambda f() takes a state and returns a heuristic function based on the provided input
            openSet = [Node([], self.initial_state, None, 0, 0, lambda state: self.run_bfs_manhattan_distance(state) if self.heuristic == "manhattan" else self.run_bfs_misplaced_tiles(state)(self.initial_state))]
            
            #Variable used to show that open set is empty but goal was never reached
            failure = None, None, None, None
            
            
            #According to the algorithm in the book, we initialize an empty set first
            fScore = set()
            
            tentative_gScore = []
    
            #Iterate while we still have elements 
            while openSet is not []:
                
                #The first element stored in the frontier
                #This operation can occur in O(Log(N)) time if openSet is a min-heap or a priority queue
                current = openSet.pop(0)
                #_, current = heapq.heappop(frontier)
    
                #Need to check if the current state = the goal state using goal_test()
                if self.goal_test(current.state):
                    
                    #Here i get the end time of the algorithm to compute the difference
                    end_time = time()
                    
                    #Difference between the end of the execution of the process and its beginning
                    diff_time = end_time - start_time
                    
                    #Virtual memory calculation
                    VM = (psutil.virtual_memory().used / pow(1024, 2))
                    #moves = current.moves[1:] + [current]
                    moves = self.reconstruct_path(current.moves[1:], current)

                    #The following values will be stored inside the solution_moves
                    return moves, nodes_expanded, diff_time, VM
        
                current_state_tuple = tuple(current.state)

                if current_state_tuple in fScore:
                    continue
                
                
                if not any([current.state == _ for _ in tentative_gScore]):
                    
                    #d(current,neighbor) is the weight of the edge from current to neighbor
                    #tentative_gScore is the distance from start to the neighbor through current
                    tentative_gScore.append(current.state)

                    # Increment number of expanded nodes
                    nodes_expanded = nodes_expanded + 1
                    
                    #Lambda f() that generates the successors of a given node for further exploration
                    explore = lambda self, node: [
                    Node(
                        node.moves + [node],
                        state,
                        action,
                        node.depth + 1,
                        node.gScore + 1,
                        
                        #Select appropriate h-function based on the provided input
                        (self.run_bfs_manhattan_distance(state) if self.heuristic == "manhattan" else self.run_bfs_misplaced_tiles(state))
                    )
                    for action, state in zip(*self.getAdjNode(node.state))
                    ]
                    
                    
                    #Allows to explore the current node, access its state, check the next node and its state.
                    #If the new neighbor node's state is not fisited, then we add it to the frontier.
                    for new_node in explore(self, current):
                        
                        #This path to neighbor is better than any previous one. Record it!
                        insort(openSet, new_node)
                
            #Open set is empty but goal was never reached
            return failure
          
            
    #Checks if the dimensions entered are okay
    def sanityCheck(self, x, y, width = 4, height = 4):
        return 0 <= x < width and 0 <= y < height

    
    #Allows to find the neighbor node of the current
    def getAdjNode(self, state):
        
        row, col = self.start_pos(state)
    
        # All possible moves that can be used to solve the puzzle
        #        U
        #        |
        # L(-1)---|---R(+1)
        #        |
        #        D
        # path = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # move_names = ['U', 'D','L', 'R']
            
        #((A, A), B) 
        path = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        move_names = ['L', 'R', 'U', 'D']
        
        #Standard list comprehension technique
        moves = [(row_offset, col_offset, A) for (row_offset, col_offset), A in zip(path, move_names)]
        
        # Standard list comprehension technique
        archive = zip(path, move_names)
        
        #Lambda function used to check the adjacent nodes for similarity
        #check_Adjacent = lambda board: (next(((i, j) for i, row in enumerate(board) for j, cell in enumerate(row) if cell == 0), None))
        #adjacents = []
        #adjacent_position = check_Adjacent(state)
    
        #if adjacent_position is not None:

        '''for (A, B), move_name in archive:
    
                # Save the original
                upd_a = a + A
                upd_b = b + B
    
                if self.sanityCheck(upd_a, upd_b, width=4, height=4):
    
                    result = [list(row) for row in state]
    
                    # Pack + unpack technique
                    result[a][b], result[upd_a][upd_b] = result[upd_a][upd_b], result[a][b]
                    actions.append(move_name)
                    states.append(result)
    
            self.memo[container] = (actions, states)
    
            return (actions, states)
        else:
            return (None, None)'''
        
        #Generates a list of adj states + actions by iterating through a set of predefined moves 
        neighbors = [
            (A, B)
            
            #Allows to move the empty state in the direction of displacement
            for x_disp, y_disp, A in moves
                
                #Checks if the dimensions entered are okay
                if(self.sanityCheck(row + x_disp, col + y_disp))
                    for B in [self.update_state(state, row, col, x_disp, y_disp)]
        ]
      
        actions, states = zip(*neighbors)
      
        return list(actions), list(states)
    
    
    #Heuristic function to compute the Manhattan distance
    def run_bfs_manhattan_distance(self, state):
        
        #print("MANHATTAN IS RUNNING!")
        #If x != 0 => filters out the tiles that are not misplaced == 0 => no contribution to manhattan dist.
        #sum() adds up all the abs differences == total Manhattan distance.
        return sum(abs(x - y) for x, y in zip(state, self.solution) if x != 0)

    
    # Heuristic function to compute the number of misplaced tiles
    def run_bfs_misplaced_tiles(self, state):
        
        #print("run_bfs_misplaced_tiles IS RUNNING!")
        goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
        misplaced = sum(1 for i in range(len(state)) if state[i] != goal_state[i])
        
        return misplaced
    
    
    #Helper function used to rectonstruct the total path according to the algorithms
    def reconstruct_path(self, came_from, current):
         
        total_path = came_from + [current]
        
        return total_path
        

    #Helper function to calculate the starting position of the tile
    def start_pos(self, state):
        
        size = self.size
        
        #Iterate over both the indices + values in the state list simultaneously
        for index, value in enumerate(state):
            if value == 0:
                row = index // size
                col = index % size
                return row, col
        return None.__class__()
 
        
    #Used to check if the solution was found and matches the provided pattern   
    def goal_test(self, cur_tiles):
        
        goal_state_def = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '0']
        goal_state = [int(_) for _ in goal_state_def]

        return cur_tiles == goal_state
    
    
    #The following function allows to update the current state of the tile on the board
    def update_state(self, state, row, col, x_disp, y_disp):
        
        #Create a shallow copy of the state
        new_state = state[:]
        
        #Size of the board => 4 by 4
        size = self.size
        
        #New row + col indices = row/col + its displacement
        new_row = row + x_disp
        new_col = col + y_disp
        
        #New + zero index calculations
        new_idx = new_row * size + new_col
        zero_idx = row * size + col
    
        #Swap the values at the new_index and zero_index positions to achieve the move => new state
        new_state[new_idx], new_state[zero_idx] = new_state[zero_idx], new_state[new_idx]
    
        return new_state

    
    #Helper function to display the required statistics
    def process_solution_moves(self, solution_moves):
        
        #Here I check if the solution is valid. If so, proceed next.
        #if solution_moves is not []:
        if solution_moves is not None:
            
            #Unpack the solution_moves container to extract the corresponding values
            path, nodes_expanded, time_taken, memory_used = solution_moves
      
            #Simply converts the number of kb achieved into bytes + casting to int 
            kb = int(round(memory_used, 0))
           
            #Obtain the path from the nodes
            path_str = ''.join([n.action for n in path])
        
            print(f"Moves: {path_str}")
            print(f"Number of expanded Nodes: {nodes_expanded}")
            print(f"Time Taken: {round(time_taken, 3)}")
            print(f"Max Memory: {kb}kb")
            return "".join(path_str) #Get the list of moves to solve the puzzle. Format is "RDLDDRR"
        else:
            print("Sorry, but the solution was not found.")
            return -1


#The entry points of the program
if __name__ == "__main__":
    
    #Initialize an object agent of the class Search
    agent = Search.solve(input("Please, enter the board configuration to start [Format: '1 0 2 4 5 7 3 8 9 6 11 12 13 10 14 15']: "), input("Choose a heuristic function to use [Format: 'manhattan' or 'misplaced tiles': "))
   
    
     
