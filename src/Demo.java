import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Demo {
	public static int bottomK = 11;

	public static void main(String[] args) throws IOException, InterruptedException {
		// Load model and data
	    
		MultiLayerNetwork vgg = ModelSerializer.restoreMultiLayerNetwork("models/vgg19.dl4jmodel");
		MultiLayerNetwork model = getBottomLayers(vgg, bottomK);

		RecordReader recordReader = new ImageRecordReader(224, 224, 3);
		recordReader.initialize(new FileSplit(new File("data")));
		RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 1);
		recordReader.close();

		if (dataIter.hasNext()) {
			INDArray data = preprocess(dataIter.next().getFeatureMatrix());
			INDArray output = model.output(data);
			System.out.println("Output size: " + output.shapeInfoToString());

			// Plot features
			int nrows = (int) Math.pow(2, Math.round(Math.log(output.size(1)) / Math.log(2)) / 2);
			FeatureVisualizer visual = new FeatureVisualizer(output.size(2), nrows, output.size(1) / nrows,
					600.0 / (output.size(2) * nrows));
			visual.visualize(output);
		} else {
			System.out.println("No input data found");
		}

	}

	public static INDArray preprocess(INDArray raw) {
		// Substract means from each channel
		return raw.sub(Nd4j.create(new double[][] { { 103.939 }, { 116.779 }, { 123.68 } })
				.broadcast(new int[] { 3, 224 * 224 }));
	}

	public static MultiLayerNetwork getBottomLayers(MultiLayerNetwork net, int k) {
		// Get bottom k layers from the pretrained model

		System.out.println("Reconstructing...");
		ListBuilder conf = new NeuralNetConfiguration.Builder().activation(Activation.RELU).list();
		k = Math.min(k, net.getnLayers());

		for (int i = 0; i < k; i++) {
			Layer layer = net.getLayer(i).conf().getLayer();
			conf.layer(i, layer);
			System.out.println("Layer " + i + ": "
					+ (layer.getClass() == SubsamplingLayer.class ? "Max-pooling" : "Convolutional") + " layer added");
		}

		if (k < net.getnLayers() && net.getLayer(k - 1).getClass() == ConvolutionLayer.class) {
			System.out.print("Layer " + k + ": Dummy output layer added");
			conf.layer(k, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.NONE, new int[] { 1, 1 })
					.name("dummy").build());
		}

		MultiLayerNetwork model = new MultiLayerNetwork(conf.backprop(false)
		        .setInputType(InputType.convolutional(224, 224, 3)).build());
		model.init();

		for (int i = 0; i < k; i++) {
			model.getLayer(i).setParams(net.getLayer(i).params());
		}

		return model;
	}
}


