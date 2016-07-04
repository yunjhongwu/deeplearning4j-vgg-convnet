
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class FeatureVisualizer extends JFrame {
	private static final long serialVersionUID = 1L;
	private BufferedImage image;
	private final int width, height, gridRows, gridCols;
	private double zoom;
	private JPanel canvas;
	private int featureSize;

	public FeatureVisualizer(int featureSize, int gridRows, int gridCols, double zoom) {

		this.gridRows = gridRows;
		this.gridCols = gridCols;
		this.featureSize = featureSize;
		this.zoom = zoom;

		this.height = gridRows * (int) (featureSize * zoom) + 1;
		this.width = gridCols * (int) (featureSize * zoom) + 1;

	}

	public void visualize(INDArray feature) {
		if (gridRows * gridCols * featureSize * featureSize != feature.length()) {
			throw new IllegalArgumentException("(" + (gridRows * gridCols) + ", " + featureSize + ", " + featureSize
					+ ") != #feautre " + feature.length());
		}

		int nrows = gridRows * featureSize;
		int ncols = gridCols * featureSize;
		INDArray matrix = Nd4j.create(new int[] { nrows, ncols });
		int numRep = gridRows * gridCols;
		for (int i = 0; i < numRep; i++) {
			int idx = i * featureSize * featureSize;
			for (int j = 0; j < featureSize; j++) {
				int rowIdx = i / gridCols * featureSize + j;
				for (int k = 0; k < featureSize; k++) {
					int colIdx = i % gridCols * featureSize + k;
					matrix.put(new int[] { rowIdx, colIdx }, feature.getScalar(idx + j * featureSize + k));
				}
			}
		}

		matrix = matrix.mul(255 / matrix.maxNumber().doubleValue());

		this.image = new BufferedImage(ncols, nrows, BufferedImage.TYPE_INT_RGB);
		for (int i = 0; i < nrows; i++) {
			for (int j = 0; j < ncols; j++) {
				int r = (int) Math.sqrt(matrix.getDouble(i, j) * 255) & 0x0ff;

				image.setRGB(j, i, (r << 16) + (r << 8) + r);
			}
		}
		view();
	}

	private void view() {
		canvas = new Canvas();
		canvas.setBackground(Color.black);
		this.setContentPane(canvas);
		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
		this.pack();
		this.setVisible(true);
	}

	private class Canvas extends JPanel {
		private static final long serialVersionUID = 1L;

		public Canvas() {
			setPreferredSize(new Dimension(width, height));
		}

		@Override
		public void paintComponent(Graphics g) {
			super.paintComponent(g);
			g.drawImage(image, 0, 0, width, height, null);
			g.setColor(Color.getHSBColor((float) 47.0 / 360, (float) 0.5, (float) 0.2));
			int squareSize = (int) (featureSize * zoom);
			for (int i = 0; i < gridRows; i++) {
				for (int j = 0; j < gridCols; j++) {
					g.drawRect(j * squareSize, i * squareSize, squareSize, squareSize);
				}
			}
			g.dispose();
		}
	}

}
