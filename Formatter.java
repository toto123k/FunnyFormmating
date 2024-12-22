package com.segali.funnyformatter;

import java.util.ArrayList;
import java.util.List;

public class Formatter {
	/**
	 * Cleans the given Java code by removing unnecessary characters.
	 *
	 * @param javaCode The original Java code.
	 * @return Cleaned Java code.
	 */
	private static String cleanCode(String javaCode) {
		return javaCode.replace("\r", "").strip();
	}

	/**
	 * Parses Java code to identify comments and their structure, ignoring
	 * anything that appears inside string literals (including multi-line text blocks).
	 *
	 * Each returned element is a String array with four items:
	 *   [0] -> The original line content
	 *   [1] -> "true" if the line is flagged as a comment, "false" otherwise
	 *   [2] -> "true" if we ended *this* line inside a string, "false" otherwise
	 *   [3] -> "true" if we ended *this* line inside a text block, "false" otherwise
	 *
	 * @param javaCode The Java code as a string.
	 * @return A list of string arrays with comment and state info.
	 */
	public static List<String[]> parseCode(String javaCode) {
		List<String[]> result = new ArrayList<>();

		StringBuilder currentLineBuffer = new StringBuilder();
		boolean inString = false;
		boolean inTextBlock = false; // for triple-quote text blocks
		boolean inMultilineComment = false;
		boolean inSingleLineComment = false;

		// Track the delimiter type for regular strings: ' or "
		char stringDelimiter = 0;

		int length = javaCode.length();

		for (int i = 0; i < length; i++) {
			char c = javaCode.charAt(i);
			char nextChar = (i + 1 < length) ? javaCode.charAt(i + 1) : '\0';

			// Check for line break
			if (c == '\n') {
				String line = currentLineBuffer.toString();
				currentLineBuffer.setLength(0); // reset for next line

				// Single-line comment ends at the newline
				if (inSingleLineComment) {
					inSingleLineComment = false;
				}

				boolean isComment = inMultilineComment || (line.strip().startsWith("//"));
				result.add(new String[] {
					line,
					String.valueOf(isComment),
					String.valueOf(inString),
					String.valueOf(inTextBlock)
				});
				continue;
			}

			// If already in single-line comment, just keep appending until '\n'
			if (inSingleLineComment) {
				currentLineBuffer.append(c);
				continue;
			}

			// If not in a string, not in text block, and not in a multiline comment,
			// check if we're starting a comment or a string or a text block
			if (!inString && !inTextBlock && !inMultilineComment) {
				// Start of single-line comment (//)
				if (c == '/' && nextChar == '/') {
					inSingleLineComment = true;
					currentLineBuffer.append(c); // append '/'
					continue;
				}

				// Start of multi-line comment (/*)
				if (c == '/' && nextChar == '*') {
					inMultilineComment = true;
					currentLineBuffer.append(c); // append '/'
					continue;
				}

				// Start of string literal (" or ')
				if (c == '"' || c == '\'') {
					inString = true;
					stringDelimiter = c;
				}

				// Start of text block triple-quote (""")
				if (c == '"' && i + 2 < length
					&& javaCode.charAt(i + 1) == '"'
					&& javaCode.charAt(i + 2) == '"') {
					inTextBlock = true;
					currentLineBuffer.append(c);
					currentLineBuffer.append(javaCode.charAt(i + 1));
					currentLineBuffer.append(javaCode.charAt(i + 2));
					i += 2;
					continue;
				}
			}
			// If inside a multiline comment, check for its end
			else if (inMultilineComment) {
				if (c == '*' && nextChar == '/') {
					inMultilineComment = false;
				}
				currentLineBuffer.append(c);
				continue;
			}
			// If inside a regular string
			else if (inString) {
				if (c == stringDelimiter) {
					// Check if it's escaped
					boolean isEscaped = false;
					int backslashCount = 0;
					int checkPos = i - 1;
					while (checkPos >= 0 && javaCode.charAt(checkPos) == '\\') {
						backslashCount++;
						checkPos--;
					}
					isEscaped = (backslashCount % 2 != 0);

					if (!isEscaped) {
						inString = false;
					}
				}
			}
			// If inside a text block, check for the closing """
			else if (inTextBlock) {
				if (c == '"' && i + 2 < length
					&& javaCode.charAt(i + 1) == '"'
					&& javaCode.charAt(i + 2) == '"') {
					inTextBlock = false;
					currentLineBuffer.append(c);
					currentLineBuffer.append(javaCode.charAt(i + 1));
					currentLineBuffer.append(javaCode.charAt(i + 2));
					i += 2;
					continue;
				}
			}

			// Add the current character
			currentLineBuffer.append(c);
		}

		// Handle trailing text (if no trailing newline)
		if (currentLineBuffer.length() > 0) {
			String line = currentLineBuffer.toString();
			boolean isComment = inSingleLineComment || inMultilineComment || line.strip().startsWith("//");
			result.add(new String[] {
				line,
				String.valueOf(isComment),
				String.valueOf(inString),
				String.valueOf(inTextBlock)
			});
		}

		return result;
	}

	/**
	 * Formats Java code with consistent spacing, structure, and removes redundant blank lines.
	 *
	 * @param originalText The Java code to format.
	 * @return The formatted Java code.
	 */
	public static String formatCode(String originalText) {
		List<String[]> parsedLines = parseCode(originalText);
		StringBuilder formattedCode = new StringBuilder();

		boolean shouldAddNewline = true;
		boolean previousLineEmpty = false; // Track consecutive blank lines

		// Iterate through the parsed lines, excluding the last one for pairwise comparison
		for (int i = 0; i < parsedLines.size() - 1; i++) {
			String currentLine    = parsedLines.get(i)[0];
			boolean isComment     = Boolean.parseBoolean(parsedLines.get(i)[1]);
			boolean endedInString = Boolean.parseBoolean(parsedLines.get(i)[2]);
			boolean endedInText   = Boolean.parseBoolean(parsedLines.get(i)[3]);

			String nextLine = parsedLines.get(i + 1)[0];

			// If this line is purely blank...
			if (currentLine.isBlank()) {
				if (!previousLineEmpty) {
					formattedCode.append("\n");
					previousLineEmpty = true;
				}
				continue;
			} else {
				previousLineEmpty = false;
			}

			// If we ended inside a string or text block, skip fancy reformatting:
			if (!isComment && !endedInString && !endedInText) {
				// 1) If current line ends with ';' and next line starts with certain keywords
				if (currentLine.endsWith(";")) {
					String nextWord = getFirstWord(nextLine).split(" ")[0];
					if (List.of("switch", "while", "for", "do", "case", "if", "return").contains(nextWord)) {
						currentLine += "\n";
					}
				}

				if (getFirstWord(currentLine).equals("}")
					&& !getFirstWord(nextLine).startsWith("}") && !currentLine.endsWith("{")) {
					String nextWord = getFirstWord(nextLine);
					if (!nextWord.equals("}") && !nextWord.isEmpty()) {
						currentLine += "\n";
					}
				}

				// 3) If current line is an opening brace '{' (or ends with '{'),
				//    and next line is empty, skip the newline
				if (getFirstWord(currentLine).endsWith("{")) {
					if (getFirstWord(nextLine).isEmpty()) {
						shouldAddNewline = false;
					}
				}
			}

			// Actually append the current line
			if (shouldAddNewline) {
				formattedCode.append(currentLine).append("\n");
			} else {
				formattedCode.append(currentLine);
				shouldAddNewline = true; // reset for the next round
			}
		}

		// Finally, add the last line
		String[] lastLineArr = parsedLines.get(parsedLines.size() - 1);
		String lastLine       = lastLineArr[0];
		boolean lastBlank     = lastLine.isBlank();

		if (!lastBlank) {
			formattedCode.append(lastLine);
		}

		// Trim trailing blank lines
		return formattedCode.toString().stripTrailing();
	}

	/**
	 * Extracts the first word from a line, ignoring leading spaces or tabs.
	 *
	 * @param line The line of code.
	 * @return The first meaningful word in the line, or an empty string if none is found.
	 */
	private static String getFirstWord(String line) {
		String stripped = line.stripLeading();
		if (stripped.isEmpty()) return "";
		return stripped.split("\\s+")[0];
	}

	public static void main(String[] args) {
		String test = """
			    /**
			       * Determines the character to display at a specific board cell.
			       *
			       * @param x The X coordinate of the cell.
			       * @param y The Y coordinate of the cell.
			       * @return The character representing the cell's content.
			       */
			      private static char getBoardCellContent(int x, int y) {
			        // Check if the cell is part of the carpet
			        if (carpetTopLeftX <= x
			            && x <= carpetTopLeftX + carpetSize - 1
			            && carpetTopLeftY <= y
			            && y <= carpetTopLeftY + carpetSize - 1) {
			          return '*';
			        }
			        // Check if the cell contains the current player
			        if (currentPlayerX == x && currentPlayerY == y) {
			          return currentPlayer;
			        }
			
			        // Check if the cell contains the other player
			        if (otherPlayerX == x && otherPlayerY == y) {
			          return otherPlayer;
			        }
			
			        // Empty cell
			        return ' ';
			      }
			""";

		System.out.println(formatCode(test));
	}
}