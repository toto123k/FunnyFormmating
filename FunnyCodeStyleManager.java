package com.segali.funnyformatter;

import com.intellij.lang.ASTNode;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.fileTypes.FileType;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.util.Computable;
import com.intellij.openapi.util.TextRange;
import com.intellij.psi.PsiElement;
import com.intellij.psi.PsiFile;
import com.intellij.psi.codeStyle.ChangedRangesInfo;
import com.intellij.psi.codeStyle.CodeStyleManager;
import com.intellij.psi.codeStyle.Indent;
import com.intellij.util.IncorrectOperationException;
import com.intellij.util.ThrowableRunnable;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Collection;

public class FunnyCodeStyleManager extends CodeStyleManager {

	private final CodeStyleManager originalManager;
	private final Project project;

	public FunnyCodeStyleManager(@NotNull Project project) {
		this.project = project;
		this.originalManager = CodeStyleManager.getInstance(project);
	}

	// ============================
	//   OVERRIDDEN FORMAT METHODS
	// ============================

	@NotNull
	@Override
	public Project getProject() {
		return project;
	}

	@NotNull
	@Override
	public PsiElement reformat(@NotNull PsiElement element) {
		PsiElement result = originalManager.reformat(element);
		doFunnyFormatting(element.getContainingFile());
		return result;
	}

	@NotNull
	@Override
	public PsiElement reformat(@NotNull PsiElement element, boolean canChangeWhiteSpacesOnly) {
		PsiElement result = originalManager.reformat(element, canChangeWhiteSpacesOnly);
		doFunnyFormatting(element.getContainingFile());
		return result;
	}

	@Override
	public PsiElement reformatRange(@NotNull PsiElement element, int startOffset, int endOffset) {
		PsiElement result = originalManager.reformatRange(element, startOffset, endOffset);
		doFunnyFormatting(element.getContainingFile());
		return result;
	}

	@Override
	public PsiElement reformatRange(@NotNull PsiElement element,
		int startOffset,
		int endOffset,
		boolean canChangeWhiteSpacesOnly) {
		PsiElement result = originalManager.reformatRange(element, startOffset, endOffset, canChangeWhiteSpacesOnly);
		doFunnyFormatting(element.getContainingFile());
		return result;
	}

	@Override
	public void reformatText(@NotNull PsiFile file, int startOffset, int endOffset) {
		originalManager.reformatText(file, startOffset, endOffset);
		doFunnyFormatting(file);
	}

	@Override
	public void reformatText(@NotNull PsiFile file, @NotNull Collection<? extends TextRange> ranges) {
		originalManager.reformatText(file, ranges);
		doFunnyFormatting(file);
	}

	@Override
	public void reformatTextWithContext(@NotNull PsiFile file, @NotNull ChangedRangesInfo changedRangesInfo) {
		originalManager.reformatTextWithContext(file, changedRangesInfo);
		doFunnyFormatting(file);
	}

	@Override
	public void adjustLineIndent(@NotNull PsiFile file, TextRange rangeToAdjust)
		throws IncorrectOperationException {
		originalManager.adjustLineIndent(file, rangeToAdjust);
	}

	@Override
	public int adjustLineIndent(@NotNull PsiFile file, int offset) throws IncorrectOperationException {
		return originalManager.adjustLineIndent(file, offset);
	}

	@Override
	public int adjustLineIndent(@NotNull Document document, int offset) {
		return originalManager.adjustLineIndent(document, offset);
	}

	@Override
	public boolean isLineToBeIndented(@NotNull PsiFile file, int offset) {
		return originalManager.isLineToBeIndented(file, offset);
	}

	@Override
	public @Nullable String getLineIndent(@NotNull PsiFile file, int offset) {
		return originalManager.getLineIndent(file, offset);
	}

	@Override
	public @Nullable String getLineIndent(@NotNull Document document, int offset) {
		return originalManager.getLineIndent(document, offset);
	}

	@Override
	public Indent getIndent(String text, FileType fileType) {
		return originalManager.getIndent(text, fileType);
	}

	@Override
	public String fillIndent(Indent indent, FileType fileType) {
		return originalManager.fillIndent(indent, fileType);
	}

	@Override
	public Indent zeroIndent() {
		return originalManager.zeroIndent();
	}

	@Override
	public void reformatNewlyAddedElement(@NotNull ASTNode block, @NotNull ASTNode addedElement)
		throws IncorrectOperationException {
		originalManager.reformatNewlyAddedElement(block, addedElement);
	}

	@Override
	public boolean isSequentialProcessingAllowed() {
		return originalManager.isSequentialProcessingAllowed();
	}

	@Override
	public void performActionWithFormatterDisabled(Runnable r) {
		originalManager.performActionWithFormatterDisabled(r);
	}

	@Override
	public <T extends Throwable> void performActionWithFormatterDisabled(ThrowableRunnable<T> r) throws T {
		originalManager.performActionWithFormatterDisabled(r);
	}

	@Override
	public <T> T performActionWithFormatterDisabled(Computable<T> r) {
		return originalManager.performActionWithFormatterDisabled(r);
	}

	// ============================
	//    CUSTOM FORMATTING LOGIC
	// ============================

	private void doFunnyFormatting(PsiFile psiFile) {
		if (psiFile == null) return;
		String originalText = psiFile.getText();
		String newText = Formatter.formatCode(originalText); // Replace with your custom formatter logic
		psiFile.getViewProvider().getDocument().setText(newText);
	}
}
