import svgwrite
from border import Border
from maze import Maze
from role import Role
from square import Square

# Define the Maze dimensions
dimensions = (4, 3)  # (width, height) in terms of number of squares

# Define the Maze
maze = Maze(
    squares=(
        Square(0, 0, 0, Border.TOP | Border.LEFT),
        Square(1, 0, 1, Border.TOP | Border.RIGHT),
        Square(2, 0, 2, Border.LEFT | Border.RIGHT, Role.EXIT),
        Square(3, 0, 3, Border.TOP | Border.RIGHT),
        Square(4, 1, 0, Border.BOTTOM | Border.LEFT | Border.RIGHT),
        Square(5, 1, 1, Border.LEFT | Border.RIGHT),
        Square(6, 1, 2, Border.BOTTOM | Border.LEFT),
        Square(7, 1, 3, Border.RIGHT),
        Square(8, 2, 0, Border.TOP | Border.LEFT, Role.ENTRANCE),
        Square(9, 2, 1, Border.BOTTOM),
        Square(10, 2, 2, Border.TOP | Border.BOTTOM),
        Square(11, 2, 3, Border.BOTTOM | Border.RIGHT),
    )
)

# SVG visualization function
def create_svg_maze(maze, dimensions, file_name="maze.svg", square_size=50, grid_line_width=0.5):
    width, height = dimensions
    dwg = svgwrite.Drawing(file_name, profile="tiny", size=(width * square_size, height * square_size))
    
    # Draw grid
    for row in range(height + 1):
        y = row * square_size
        dwg.add(dwg.line(start=(0, y), end=(width * square_size, y), stroke="gray", stroke_width=grid_line_width))
    for col in range(width + 1):
        x = col * square_size
        dwg.add(dwg.line(start=(x, 0), end=(x, height * square_size), stroke="gray", stroke_width=grid_line_width))

    # Draw maze squares
    for square in maze.squares:
        x, y = square.column * square_size, square.row * square_size
        
        # Draw the square's borders
        if square.border & Border.TOP:
            dwg.add(dwg.line(start=(x, y), end=(x + square_size, y), stroke="black", stroke_width=2))
        if square.border & Border.BOTTOM:
            dwg.add(dwg.line(start=(x, y + square_size), end=(x + square_size, y + square_size), stroke="black", stroke_width=2))
        if square.border & Border.LEFT:
            dwg.add(dwg.line(start=(x, y), end=(x, y + square_size), stroke="black", stroke_width=2))
        if square.border & Border.RIGHT:
            dwg.add(dwg.line(start=(x + square_size, y), end=(x + square_size, y + square_size), stroke="black", stroke_width=2))
        
        # Highlight Entrance and Exit
        if square.role == Role.ENTRANCE:
            dwg.add(dwg.rect(insert=(x, y), size=(square_size, square_size), fill="green", opacity=0.3))
            dwg.add(dwg.text("E", insert=(x + square_size / 2, y + square_size / 1.5), text_anchor="middle", font_size=20, fill="black"))
        elif square.role == Role.EXIT:
            dwg.add(dwg.rect(insert=(x, y), size=(square_size, square_size), fill="red", opacity=0.3))
            dwg.add(dwg.text("X", insert=(x + square_size / 2, y + square_size / 1.5), text_anchor="middle", font_size=20, fill="black"))

    # Save the SVG file
    dwg.save()
    print(f"SVG maze saved to {file_name}")

# Generate the SVG
create_svg_maze(maze, dimensions, file_name="maze_visualization_with_grid.svg")
