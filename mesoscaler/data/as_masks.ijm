Dialog.create("As masks...");
Dialog.addNumber("Width", 512);
Dialog.addNumber("Height", 512);
Dialog.addString("Prefix", "Mask_");
Dialog.addDirectory("Directory", File.getDefaultDir);
Dialog.show();

maskWidth = Dialog.getNumber();
maskHeight = Dialog.getNumber();
basePrefix = Dialog.getString();
saveDir    = Dialog.getString();

numROIs = RoiManager.size;
for (i = 0; i < numROIs; i++) {
	maskName = basePrefix + RoiManager.getName(i);
	newImage(maskName, "8-bit black", maskWidth, maskHeight, 1);
	RoiManager.select(i);
	run("Fill", "slice");
	run("Select None");
	saveAs("PNG", saveDir + File.separator + maskName + ".png");
	close();
}


