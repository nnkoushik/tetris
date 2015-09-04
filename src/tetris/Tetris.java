package tetris;

import java.io.*;
import java.util.*;

public class Tetris {
	int gameBoardRows;
	int gameBoardColumns;
	
	class GameState {
		boolean[][] gameBoard;
		
		GameState() {
			gameBoard = new boolean[gameBoardRows][gameBoardColumns];
			
			for(int i = 0; i < gameBoardRows; ++ i)
				for(int j = 0; j < gameBoardColumns; ++ j)
					gameBoard[i][j] = false;
		}
		
		GameState(GameState original) {
			gameBoard = new boolean[gameBoardRows][gameBoardColumns];
			
			for(int i = 0; i < gameBoardRows; ++ i)
				for(int j = 0; j < gameBoardColumns; ++ j)
					gameBoard[i][j] = original.gameBoard[i][j];
		}
		
		@Override
		public String toString() {
			StringBuilder stringBuilder = new StringBuilder();
			
			for(int i = 0; i < gameBoardRows; ++ i) {
				for(int j = 0; j < gameBoardColumns; ++ j)
					stringBuilder.append(gameBoard[i][j] ? "X " : ". ");
				
				stringBuilder.append("\n");
			}
			
			return stringBuilder.toString();
		}
		
		@Override
		public int hashCode() {
			return toString().hashCode();
		}
		
		@Override
		public boolean equals(Object other) {
			if(!(other instanceof GameState)) {
				return false;
			}
			
			for(int i = 0; i < gameBoardRows; ++ i)
				for(int j = 0; j < gameBoardColumns; ++ j)
					if(this.gameBoard[i][j] != ((GameState)other).gameBoard[i][j])
						return false;
			
			return true;
		}
		
		public void set(int row, int column, boolean bool) {
			gameBoard[row][column] = bool;
		}
		
		int columnHeight(int column) {
			int height;
			for(height = 0; (height < gameBoardRows) && !gameBoard[height][column]; ++ height);
			return height;
		}

		boolean fullRow(int row) {
			for(int j = 0; j < gameBoardColumns; ++ j)
				if(!gameBoard[row][j]) return false;

			return true;
		}

		int clearRow(int row) {
			if(fullRow(row)) {
				for(int j = 0; j < gameBoardColumns; ++ j) {
					for(int i = row; row > 0; -- row)
						gameBoard[i][j] = gameBoard[i - 1][j];
					gameBoard[0][j] = false;
				}
				
				return 1;
			}
			
			return 0;
		}
		
		public GameState insertIntoGameBoard(PieceState pieceState){
			GameState newGameState = new GameState(this);
			
			if((pieceState.piece == Piece.S) && (pieceState.orientation == Orientation.VERTICAL)) {
				int h1 = columnHeight(pieceState.column);
				int h2 = columnHeight(pieceState.column + 1);
				
				if(h2 <= h1) {
					if(h2 < 3) return null;
					
					newGameState.set(h2 - 3, pieceState.column, true);
					newGameState.set(h2 - 2, pieceState.column, true);
					newGameState.set(h2 - 2, pieceState.column + 1, true);
					newGameState.set(h2 - 1, pieceState.column + 1, true);
					
					newGameState.clearRow(newGameState.clearRow(h2 - 1) + h2 - 2);
				} else {
					if(h1 < 2) return null;
					
					newGameState.set(h1 - 2, pieceState.column, true);
					newGameState.set(h1 - 1, pieceState.column, true);
					newGameState.set(h1 - 1, pieceState.column + 1, true);
					newGameState.set(h1, pieceState.column + 1, true);
					
					newGameState.clearRow(newGameState.clearRow(h1) + h1 - 1);
				}
			}
			
			if((pieceState.piece == Piece.Z) && (pieceState.orientation == Orientation.VERTICAL)) {
				int h1 = columnHeight(pieceState.column);
				int h2 = columnHeight(pieceState.column + 1);
				
				if(h1 <= h2) {
					if(h1 < 3) return null;
					
					newGameState.set(h1 - 3, pieceState.column + 1, true);
					newGameState.set(h1 - 2, pieceState.column + 1, true);
					newGameState.set(h1 - 2, pieceState.column, true);
					newGameState.set(h1 - 1, pieceState.column, true);
					
					newGameState.clearRow(newGameState.clearRow(h1 - 1) + h1 - 2);
				} else {
					if(h2 < 2) return null;
					
					newGameState.set(h2 - 2, pieceState.column + 1, true);
					newGameState.set(h2 - 1, pieceState.column + 1, true);
					newGameState.set(h2 - 1, pieceState.column, true);
					newGameState.set(h2, pieceState.column, true);
					
					newGameState.clearRow(newGameState.clearRow(h2) + h2 - 1);
				}
			}
			
			if((pieceState.piece == Piece.S) && (pieceState.orientation == Orientation.HORIZONTAL)) {
				int h1 = columnHeight(pieceState.column);
				int h2 = columnHeight(pieceState.column + 1);
				int h3 = columnHeight(pieceState.column + 2);
				
				if(h1 <= h3 || h2 <= h3) {
					int min = (h1 < h2) ? h1 : h2;
					
					if(min < 2) return null;
					
					newGameState.set(min - 1, pieceState.column, true);
					newGameState.set(min - 1, pieceState.column + 1, true);
					newGameState.set(min - 2, pieceState.column + 1, true);
					newGameState.set(min - 2, pieceState.column + 2, true);
					
					newGameState.clearRow(min - 1);
				} else {
					if(h3 < 1) return null;
					
					newGameState.set(h3, pieceState.column, true);
					newGameState.set(h3, pieceState.column + 1, true);
					newGameState.set(h3 - 1, pieceState.column + 1, true);
					newGameState.set(h3 - 1, pieceState.column + 2, true);
					
					newGameState.clearRow(h3);
				}
			}

			if((pieceState.piece == Piece.Z) && (pieceState.orientation == Orientation.HORIZONTAL)) {
				int h1 = columnHeight(pieceState.column);
				int h2 = columnHeight(pieceState.column + 1);
				int h3 = columnHeight(pieceState.column + 2);
				
				if(h2 <= h1 || h3 <= h1) {
					int min = (h2 < h3) ? h2 : h3;
					
					if(min < 2) return null;
					
					newGameState.set(min - 2, pieceState.column, true);
					newGameState.set(min - 2, pieceState.column + 1, true);
					newGameState.set(min - 1, pieceState.column + 1, true);
					newGameState.set(min - 1, pieceState.column + 2, true);
					
					newGameState.clearRow(min - 1);
				} else {
					if(h1 < 1) return null;
					
					newGameState.set(h1 - 1, pieceState.column, true);
					newGameState.set(h1 - 1, pieceState.column + 1, true);
					newGameState.set(h1, pieceState.column + 1, true);
					newGameState.set(h1, pieceState.column + 2, true);
					
					newGameState.clearRow(h1);
				}
			}
			
			return newGameState;
		}
	}
	
	public Set<GameState> generatedGameStates = new HashSet<GameState>();
	
	int numberOfGames;
	int iterationsPerGame;
	
	enum Piece {
		S, Z
	}

	Piece pieceFromInt(int x) {
		switch(x) {
		case 0: return Piece.S;
		case 1: return Piece.Z;
		default: return null;
		}
	}
	
	enum Orientation {
		HORIZONTAL, VERTICAL
	}
	
	Orientation orientationFromInt(int x) {
		switch(x) {
		case 0: return Orientation.HORIZONTAL;
		case 1: return Orientation.VERTICAL;
		default: return null;
		}
	}
	
	class PieceState {
		Piece piece;
		Orientation orientation;
		int column;
		
		public PieceState(Piece piece, Orientation orientation, int column) {
			this.piece = piece;
			this.orientation = orientation;
			this.column = column;
		}
	}
	
	PieceState generateRandomPieceState(Random random) {
		Piece piece = pieceFromInt(random.nextInt(2));
		Orientation orientation = orientationFromInt(random.nextInt(2));
		int column = 0;
		
		switch(orientation) {
		case HORIZONTAL:
			column = random.nextInt(gameBoardColumns - 2);
			break;
		case VERTICAL:
			column = random.nextInt(gameBoardColumns - 1);
			break;
		}
		
		return new PieceState(piece, orientation, column);
	}
	
	Tetris(int gameBoardRows, int gameBoardColumns, int numberOfGames, int iterationsPerGame) {
		this.gameBoardRows = gameBoardRows;
		this.gameBoardColumns = gameBoardColumns;
		
		this.numberOfGames = numberOfGames;
		this.iterationsPerGame = iterationsPerGame;
	}

	void generateGameStates() {
		Random random = new Random();
		
		for(int i = 0; i < numberOfGames; ++ i) {
			GameState gameState = new GameState();
			
			for(int j = 0; j < iterationsPerGame; ++ j) {
				PieceState pieceState = generateRandomPieceState(random);
				GameState nextGameState = gameState.insertIntoGameBoard(pieceState);
				
				if(nextGameState == null) {
					break;
				}
				
				if(!generatedGameStates.contains(gameState))
					generatedGameStates.add(gameState);
				
				gameState = nextGameState;
			}
		}
	}
	
	public static void main(String[] args) throws IOException {
		int gameBoardRows = Integer.parseInt(args[0]);
		int gameBoardColumns = Integer.parseInt(args[1]);
		
		int numberOfGames = Integer.parseInt(args[2]);
		int iterationsPerGame = Integer.parseInt(args[3]);
		
		String file = args[4];
		
		Tetris tetris = new Tetris(gameBoardRows, gameBoardColumns, numberOfGames, iterationsPerGame);
		tetris.generateGameStates();
		
		BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file));
		
		for(GameState gameState : tetris.generatedGameStates) {
			bufferedWriter.write(gameState.toString());
			bufferedWriter.write("\n");
		}
		
		bufferedWriter.close();
		System.out.println(tetris.generatedGameStates.size());
	}
}
