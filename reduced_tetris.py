# Modified from Tetromino by Al Sweigart al@inventwithpython.com
# http://inventwithpython.com/pygame
# Released under a "Simplified BSD" license

import random
import time
import pygame
import sys
from copy import deepcopy
from pygame.locals import *
from hueristic_ai import HueristicAI

FPS = 25
WINDOWWIDTH = 640
WINDOWHEIGHT = 480
BOXSIZE = 20
BOARDWIDTH = 6
BOARDHEIGHT = 10
BLANK = '.'

MOVESIDEWAYSFREQ = 0.15
MOVEDOWNFREQ = 0.1

XMARGIN = int((WINDOWWIDTH - BOARDWIDTH * BOXSIZE) / 2)
TOPMARGIN = WINDOWHEIGHT - (BOARDHEIGHT * BOXSIZE) - 5

#               R    G    B
WHITE = (255, 255, 255)
GRAY = (185, 185, 185)
BLACK = (0,   0,   0)
RED = (155,   0,   0)
LIGHTRED = (175,  20,  20)
GREEN = (0, 155,   0)
LIGHTGREEN = (20, 175,  20)
BLUE = (0,   0, 155)
LIGHTBLUE = (20,  20, 175)
YELLOW = (155, 155,   0)
LIGHTYELLOW = (175, 175,  20)

BORDERCOLOR = BLUE
BGCOLOR = BLACK
TEXTCOLOR = WHITE
TEXTSHADOWCOLOR = GRAY
COLORS = (BLUE,      GREEN,      RED,      YELLOW)
LIGHTCOLORS = (LIGHTBLUE, LIGHTGREEN, LIGHTRED, LIGHTYELLOW)
assert len(COLORS) == len(LIGHTCOLORS)  # each color must have light color

TEMPLATEWIDTH = 2
TEMPLATEHEIGHT = 2

SHAPE_TEMPLATE_1 = [['O.',
                     '..'],
                    ['..',
                     'O.'],
                    ['..',
                     '.O'],
                    ['.O',
                     '..']]

SHAPE_TEMPLATE_2 = [['OO',
                     'OO']]

SHAPE_TEMPLATE_3 = [['OO',
                     '..'],
                    ['O.',
                     'O.'],
                    ['..',
                     'OO'],
                    ['.O',
                     '.O']]

SHAPE_TEMPLATE_4 = [['.O',
                     'O.'],
                    ['O.',
                     '.O']]

SHAPE_TEMPLATE_5 = [['OO',
                     'O.'],
                    ['O.',
                     'OO'],
                    ['.O',
                     'OO'],
                    ['OO',
                     '.O']]


PIECES = {'1': SHAPE_TEMPLATE_1,
          '2': SHAPE_TEMPLATE_2,
          '3': SHAPE_TEMPLATE_3,
          '4': SHAPE_TEMPLATE_4,
          '5': SHAPE_TEMPLATE_5}

class GameState:

    """

        Game state code

    """

    def __init__(self):
        global FPSCLOCK, DISPLAYSURF, BASICFONT, BIGFONT
        pygame.init()
        FPSCLOCK = pygame.time.Clock()
        DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
        BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
        BIGFONT = pygame.font.Font('freesansbold.ttf', 100)
        pygame.display.set_caption('Tetromino')

        # DEBUG
        self.total_lines = 0

        # setup variables for the start of the game
        self.board = self.getBlankBoard()
        self.movingDown = False  # note: there is no movingUp variable
        self.movingLeft = False
        self.movingRight = False
        self.score = 0
        self.lines = 0
        self.height = 0
        self.level, self.fallFreq = self.calculateLevelAndFallFreq()

        self.fallingPiece = self.getNewPiece()
        self.nextPiece = self.getNewPiece()

        self.frame_step(0)

        pygame.display.update()

    def reinit(self):

        self.board = self.getBlankBoard()
        self.movingDown = False  # note: there is no movingUp variable
        self.movingLeft = False
        self.movingRight = False
        self.score = 0
        self.lines = 0
        self.height = 0
        self.level, self.fallFreq = self.calculateLevelAndFallFreq()

        self.fallingPiece = self.getNewPiece()
        self.nextPiece = self.getNewPiece()

        self.frame_step(0)

        pygame.display.update()

    def calculateLevelAndFallFreq(self):
        # Based on the self.score, return the self.level the player is on and
        # how many seconds pass until a falling piece falls one space.
        self.level = min(int(self.lines / 10) + 1, 10)
        self.fallFreq = 0.27 - (self.level * 0.02)
        return self.level, self.fallFreq

    """ 
    
    Deep Q-Learning Code

    """

    def action_to_input(self, action):
        if action == "moveLeft":
            return 1
        elif action == "moveRight":
            return 3
        elif action == "rotateForward":
            return 2
        elif action == "rotateBackward":
            return 5
        elif action == "drop":
            return 4
        elif action == "nothing":
            return 0

    def display(self, board):
        for y in range(BOARDHEIGHT):
            items = []
            for x in range(BOARDWIDTH):
                items.append(board[x][y])

            print "".join(map(str, items))

    def get_board_image(self):
        new_board = deepcopy(self.board)

        for x in range(BOARDWIDTH):
            for y in range(BOARDHEIGHT):
                if new_board[x][y] == ".":
                    new_board[x][y] = 0
                else:
                    new_board[x][y] = 1
       
        if self.fallingPiece != None:
            for x in range(TEMPLATEWIDTH):
                for y in range(TEMPLATEHEIGHT):
                    if PIECES[self.fallingPiece['shape']][self.fallingPiece['rotation']][y][x] != BLANK:
                        new_board[x + self.fallingPiece['x']][y + self.fallingPiece['y']] = 2

        return new_board

    def frame_step(self, input):
        reward = 0
        terminal = False

        if self.fallingPiece == None:
            # No falling piece in play, so start a new piece at the top
            self.fallingPiece = self.nextPiece
            self.nextPiece = self.getNewPiece()

            # Game over
            if not self.isValidPosition(self.board, self.fallingPiece):
                terminal = True

                board_image = self.get_board_image()

                reward = -5
                
                self.reinit()
                return board_image, reward, terminal

        if input == 1:
            self.moveLeft(self.board, self.fallingPiece)
        elif input == 3:
            self.moveRight(self.board, self.fallingPiece)
        elif input == 2:
            self.rotateForward(self.board, self.fallingPiece)
        elif input == 5:
            self.rotateBackward(self.board, self.fallingPiece)
        elif (input == 4):
            self.movingDown = False
            self.movingLeft = False
            self.movingRight = False

            self.drop(self.board, self.fallingPiece)

        # Add to board if it is "fallen"
        cleared = 0
        if not self.isValidPosition(self.board, self.fallingPiece, adjY=1):
            self.addToBoard(self.board, self.fallingPiece)

            cleared = self.removeCompleteLines(self.board)
            if cleared > 0:
                if cleared == 1:
                    self.score += 40 * self.level
                elif cleared == 2:
                    self.score += 100 * self.level
                elif cleared == 3:
                    self.score += 300 * self.level
                elif cleared == 4:
                    self.score += 1200 * self.level

            self.score += self.fallingPiece['y']

            self.lines += cleared
            self.total_lines += cleared

            reward = self.height - self.get_aggregate_height(self.board) + cleared
            self.height = self.get_aggregate_height(self.board)

            self.level, self.fallFreq = self.calculateLevelAndFallFreq()
            self.fallingPiece = None

        else:
            # piece did not land, just move the piece down
            self.fallingPiece['y'] += 1

        # drawing everything on the screen
        DISPLAYSURF.fill(BGCOLOR)
        self.drawBoard()
        self.drawStatus()
        self.drawNextPiece()
        if self.fallingPiece != None:
           self.drawPiece(self.fallingPiece)

        pygame.display.update()

        if cleared > 0:
            reward = 100 * cleared

        board_image = self.get_board_image()
        return board_image, reward, terminal

    """

    BOARD OPERATIONS

    """

    def moveLeft(self, board, fallingPiece):
        if not self.isValidPosition(board, fallingPiece, adjX=-1):
            return False
        
        fallingPiece['x'] -= 1
        return True

    def moveRight(self, board, fallingPiece):
        if not self.isValidPosition(board, fallingPiece, adjX=1):
            return False

        fallingPiece['x'] += 1
        return True

    def drop(self, board, fallingPiece):
        for i in range(0, BOARDHEIGHT):
            if not self.isValidPosition(board, fallingPiece, adjY=i):
                break
        fallingPiece['y'] += i - 1

    def rotateForward(self, board, fallingPiece):
        fallingPiece['rotation'] = (
            fallingPiece['rotation'] + 1) % len(PIECES[fallingPiece['shape']])

        if not self.isValidPosition(board, fallingPiece):
            fallingPiece['rotation'] = (
                fallingPiece['rotation'] - 1) % len(PIECES[fallingPiece['shape']])
            return False

        return True

    def rotateBackward(self, board, fallingPiece):
        fallingPiece['rotation'] = (
            fallingPiece['rotation'] - 1) % len(PIECES[fallingPiece['shape']])
        if not self.isValidPosition(board, fallingPiece):
            fallingPiece['rotation'] = (
                fallingPiece['rotation'] + 1) % len(PIECES[fallingPiece['shape']])
            return False
        
        return True

    def getNewPiece(self):
        # return a random new piece in a random rotation and color
        shape = random.choice(list(PIECES.keys()))
        newPiece = {'shape': shape,
                    'rotation': random.randint(0, len(PIECES[shape]) - 1),
                    'x': int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2),
                    'y': -1,  # start it above the self.board (i.e. less than 0)
                    'color': random.randint(0, len(COLORS) - 1),
                    'num_rotations': len(PIECES[shape])}
        
        return newPiece

    def addToBoard(self, board, fallingPiece):
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if PIECES[fallingPiece['shape']][fallingPiece['rotation']][y][x] != BLANK:
                    board[x + fallingPiece['x']][y + fallingPiece['y']] = fallingPiece['color']

    def getBlankBoard(self):
        # create and return a new blank self.board data structure
        board = []
        for i in range(BOARDWIDTH):
            board.append([BLANK] * BOARDHEIGHT)
        return board

    def isOnBoard(self, x, y):
        return x >= 0 and x < BOARDWIDTH and y < BOARDHEIGHT

    def isValidPosition(self, board, fallingPiece, adjX=0, adjY=0):
        # Return True if the piece is within the self.board and not colliding
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                isAboveBoard = y + fallingPiece['y'] + adjY < 0

                if isAboveBoard and (x + fallingPiece['x'] + adjX < 0 or x + fallingPiece['x'] + adjX >= BOARDWIDTH):
                    return False

                if isAboveBoard or PIECES[fallingPiece['shape']][fallingPiece['rotation']][y][x] == BLANK:
                    continue
                if not self.isOnBoard(x + fallingPiece['x'] + adjX, y + fallingPiece['y'] + adjY):
                    return False
                if board[x + fallingPiece['x'] + adjX][y + fallingPiece['y'] + adjY] != BLANK:
                    return False
        return True

    def isCompleteLine(self, board, y):
        for x in range(BOARDWIDTH):
            if board[x][y] == BLANK:
                return False
        return True

    def removeCompleteLines(self, board):
        # Remove any completed lines on the self.board, move everything above
        # them down, and return the number of complete lines.
        numLinesRemoved = 0
        y = BOARDHEIGHT - 1  # start y at the bottom of the self.board
        while y >= 0:
            if self.isCompleteLine(board, y):
                # Remove the line and pull boxes down by one line.
                for pullDownY in range(y, 0, -1):
                    for x in range(BOARDWIDTH):
                        board[x][pullDownY] = board[x][pullDownY - 1]
                # Set very top line to blank.
                for x in range(BOARDWIDTH):
                    board[x][0] = BLANK
                numLinesRemoved += 1
                # Note on the next iteration of the loop, y is the same.
                # This is so that if the line that was pulled down is also
                # complete, it will be removed.
            else:
                y -= 1  # move on to check next row up
        return numLinesRemoved

    """

        REWARD FUNCTIONS

    """

    def get_num_completed(self, board):
        num_completed = 0

        for y in range(0, BOARDHEIGHT):
            if self.isCompleteLine(board, y):
                num_completed += 1

        return num_completed

    def is_hole(self, board, x, y):
        if board[x][y] != BLANK:
            return False

        for y2 in xrange(y - 1, -1, -1):
            if board[x][y2] != BLANK:
                return True

        return False

    def get_num_holes(self, board):
        num_holes = 0

        for y in range(0, BOARDHEIGHT):
            for x in range(0, BOARDWIDTH):
                if self.is_hole(board, x, y):
                    num_holes += 1

        return num_holes

    def get_bumpiness(self, board):
        bumpiness = 0
        heights = []
        columns_used = []

        for x in range(BOARDWIDTH):
            for y in range(BOARDHEIGHT):
                if x in columns_used:
                    continue

                if board[x][y] != BLANK:
                    heights.append(BOARDHEIGHT - y)
                    columns_used.append(x)

            if x not in columns_used:
                heights.append(0)
                columns_used.append(x)

        for i in range(0, len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])

        return bumpiness

    def get_aggregate_height(self, board):
        aggregate_height = 0
        columns_used = []

        for y in range(BOARDHEIGHT):
            for x in range(BOARDWIDTH):
                if x in columns_used:
                    continue

                if board[x][y] != BLANK:
                    aggregate_height += BOARDHEIGHT - y
                    columns_used.append(x)

        return aggregate_height

    """

        PyGame Drawing Code
    
    
    """

    def makeTextObjs(self, text, font, color):
        surf = font.render(text, True, color)
        return surf, surf.get_rect()

    def convertToPixelCoords(self, boxx, boxy):
        # Convert the given xy coordinates of the self.board to xy
        # coordinates of the location on the screen.
        return (XMARGIN + (boxx * BOXSIZE)), (TOPMARGIN + (boxy * BOXSIZE))

    def drawBox(self, boxx, boxy, color, pixelx=None, pixely=None):
        # draw a single box (each tetromino piece has four boxes)
        # at xy coordinates on the self.board. Or, if pixelx & pixely
        # are specified, draw to the pixel coordinates stored in
        # pixelx & pixely (this is used for the "Next" piece).
        if color == BLANK:
            return
        if pixelx == None and pixely == None:
            pixelx, pixely = self.convertToPixelCoords(boxx, boxy)
        pygame.draw.rect(
            DISPLAYSURF, COLORS[color], (pixelx + 1, pixely + 1, BOXSIZE - 1, BOXSIZE - 1))
        pygame.draw.rect(
            DISPLAYSURF, LIGHTCOLORS[color], (pixelx + 1, pixely + 1, BOXSIZE - 4, BOXSIZE - 4))

    def drawBoard(self):
        # draw the border around the self.board
        pygame.draw.rect(DISPLAYSURF, BORDERCOLOR, (XMARGIN - 3, TOPMARGIN - 7,
                                                    (BOARDWIDTH * BOXSIZE) + 8, (BOARDHEIGHT * BOXSIZE) + 8), 5)

        # fill the background of the self.board
        pygame.draw.rect(DISPLAYSURF, BGCOLOR, (XMARGIN, TOPMARGIN,
                                                BOXSIZE * BOARDWIDTH, BOXSIZE * BOARDHEIGHT))
        # draw the individual boxes on the self.board
        for x in range(BOARDWIDTH):
            for y in range(BOARDHEIGHT):
                self.drawBox(x, y, self.board[x][y])

    def drawStatus(self):
        # draw the self.score text
        scoreSurf = BASICFONT.render(
            'self.score: %s' % self.score, True, TEXTCOLOR)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (WINDOWWIDTH - 150, 20)
        DISPLAYSURF.blit(scoreSurf, scoreRect)

        # draw the self.level text
        levelSurf = BASICFONT.render(
            'self.level: %s' % self.level, True, TEXTCOLOR)
        levelRect = levelSurf.get_rect()
        levelRect.topleft = (WINDOWWIDTH - 150, 50)
        DISPLAYSURF.blit(levelSurf, levelRect)

    def drawPiece(self, piece, pixelx=None, pixely=None):
        shapeToDraw = PIECES[piece['shape']][piece['rotation']]
        if pixelx == None and pixely == None:
            # if pixelx & pixely hasn't been specified, use the location stored
            # in the piece data structure
            pixelx, pixely = self.convertToPixelCoords(piece['x'], piece['y'])

        # draw each of the boxes that make up the piece
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if shapeToDraw[y][x] != BLANK:
                    self.drawBox(
                        None, None, piece['color'], pixelx + (x * BOXSIZE), pixely + (y * BOXSIZE))

    def drawNextPiece(self):
        # draw the "next" text
        nextSurf = BASICFONT.render('Next:', True, TEXTCOLOR)
        nextRect = nextSurf.get_rect()
        nextRect.topleft = (WINDOWWIDTH - 120, 80)
        DISPLAYSURF.blit(nextSurf, nextRect)
        # draw the "next" piece
        self.drawPiece(self.nextPiece, pixelx=WINDOWWIDTH - 120, pixely=100)

if __name__ == '__main__':
    game_state = GameState()

    ai = HueristicAI(game_state, -0.153, 0.605, -0.645, -0.442)
    for i in range(1000):
        if game_state.fallingPiece:
            best_piece, best_actions, score = ai.get_piece_move(game_state.board, [game_state.fallingPiece, game_state.nextPiece])

            for best_action in best_actions:
                game_state.frame_step(best_action)
                #time.sleep(.1)

            game_state.frame_step(0)
        else: 
            game_state.frame_step(0)
        #time.sleep(.1)
