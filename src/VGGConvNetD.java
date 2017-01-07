
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.activations.Activation;

public class VGGConvNetD {
	public static MultiLayerConfiguration conf() {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().activation(Activation.RELU).list()
				.layer(0,
						new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nIn(3).nOut(64)
								.build())
				.layer(1, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(64).build())
				.layer(2,
						new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
								.stride(2, 2).build())
				.layer(3, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(128).build())
				.layer(4, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(128).build())
				.layer(5,
						new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
								.stride(2, 2).build())
				.layer(6, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(256).build())
				.layer(7, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(256).build())
				.layer(8, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(256).build())
				.layer(9,
						new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
								.stride(2, 2).build())
				.layer(10, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(512).build())
				.layer(11, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(512).build())
				.layer(12, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(512).build())
				.layer(13,
						new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
								.stride(2, 2).build())
				.layer(14, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(512).build())
				.layer(15, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(512).build())
				.layer(16, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).nOut(512).build())
				.layer(17, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
						.stride(2, 2).build())
				.backprop(false).setInputType(InputType.convolutional(224, 224, 3)).build();

		return conf;
	}

}
