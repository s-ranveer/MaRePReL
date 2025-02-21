import pygame
import numpy as np
import os 
import glob

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE =  (128, 0, 128) # For Agents office
BROWN = (139,69,19) # For Coffee
ORANGE = (255,165,0) # Orange for mail
YELLOW = (255, 255, 0)
LIGHT_RED = (255, 200, 200)  # Light red for agent outline
GREY = (169,169,169)
# Input string representing the office world
file_path = os.path.realpath(__file__).split('/')[:-1]
file_path.append('assets')
assets = '/'.join(file_path)

COFFEE = pygame.image.load(f"{assets}/coffee.png")  # Replace 'path_to_your_image.png' with the actual image file path
MAIL = pygame.image.load(f"{assets}/mail.png")
OFFICE = pygame.image.load(f"{assets}/office.png")
PLANT = pygame.image.load(f"{assets}/plant.png")
# Define the color mapping for characters
color_mapping = {
    ' ': WHITE,
    '+': BLACK,
    '-': BLACK,
    '|': BLACK,
    'a': GREY,
    'b': GREY,
    'c': GREY,
    'd': GREY,
    'e': ORANGE,
    'f': BROWN,
    'g': PURPLE,
    '1': YELLOW,
    '2': YELLOW,
    '3': YELLOW,
    '4': YELLOW,
    '0': YELLOW,

    '*': GREEN,
}

letters = ['a','b','c','d']
numbers = ['1','2','3','4','0']
letter_pos =  {a:[] for a in letters}

number_pos = {n:[] for n in numbers}

# possible tasks = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [7, 8], [0, 1, 2, 3],
#          [0, 1, 2, 3, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]]


def render_grid(input_string, figure_size, images=True, wait_time=100):
    
    grid = []
    assert isinstance(figure_size,int), "Aspect Ratio is 1. Please provide a single integer for screen size"
    if type(figure_size) is int:
         figure_size = (figure_size,figure_size)

    screen_width, screen_height = int(figure_size[0]*0.8),int(figure_size[1]*0.8)
    rows = input_string.strip().split('\n')
    for j, row in enumerate(rows):  # Exclude the first and last lines (the border)
        row_colors = []
        for i, char in enumerate(row):  # Exclude the first and last characters (the borders)
            color = color_mapping.get(char, WHITE)
            row_colors.append(color)
            if(char in letters):
                letter_pos[char] = [i,j]
            if (char in numbers):
                number_pos[char] = [i,j]
        grid.append(row_colors)

    # Pygame setup
    
    # Calculate cell dimensions
    cell_width = screen_width // len(grid[0])
    cell_height = screen_height // len(grid)

    #Font Size
    font_size_visit = min(int(cell_height*0.8) , int(cell_width*0.8))
    font = pygame.font.Font(None, font_size_visit)
    #Create surface
    surf = pygame.Surface(figure_size)
    surf.fill(BLACK)

    #Render the grid
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if(color==BROWN and images):
                coffee_cell = pygame.transform.scale(COFFEE, (cell_width, cell_height))
                surf.blit(coffee_cell, (x*cell_width,y*cell_height))
    
            elif(color==ORANGE and images):
                mail_cell = pygame.transform.scale(MAIL, (cell_width, cell_height))
                surf.blit(mail_cell, (x*cell_width,y*cell_height))
    
            elif(color==PURPLE and images):
                office_cell = pygame.transform.scale(OFFICE, (cell_width, cell_height))
                surf.blit(office_cell, (x*cell_width,y*cell_height))
    
            elif(color==GREEN and images):
                plant_cell = pygame.transform.scale(PLANT, (cell_width, cell_height))
                surf.blit(plant_cell, (x*cell_width,y*cell_height))
    
            else:
                pygame.draw.rect(surf, color, (x * cell_width, y * cell_height, cell_width, cell_height))
    
            # Highlight the agent (0) with a light red outline
            if color == YELLOW:
                pygame.draw.rect(surf, LIGHT_RED, (x * cell_width, y * cell_height, cell_width, cell_height), 3)

    for number,cell in number_pos.items():
        if(len(cell)>0):
            pos_x = cell[0]*cell_width + cell_width//2
            pos_y = cell[1]*cell_height + cell_height//2
            pos = (pos_x, pos_y)
            text = font.render(number, True, BLACK)
            surf.blit(text, pos)

    for letter, cell in letter_pos.items():
        pos_x = cell[0]*cell_width + cell_width//2
        pos_y = cell[1]*cell_height + cell_height//2
        pos = (pos_x, pos_y)
        text = font.render(letter.title(), True, BLACK)
        surf.blit(text, pos)

    pygame.time.wait(wait_time)
    
    return surf

def add_text(text, text_type, size): 
    
    if(text_type=="title"):  
        title_font_style = "liberationmono"   
        font_size_title = int(size*0.08)   
        title_font = pygame.font.Font(pygame.font.match_font(title_font_style), font_size_title)
        output= title_font.render(text, True, WHITE)
    
    elif(text_type=="body"):    
        game_font_style = "dejavusans"
        font_size_game_info = int(size*0.018)    
        game_font = pygame.font.Font(pygame.font.match_font(game_font_style), font_size_game_info)
        output= game_font.render(text, True, WHITE)

    return output