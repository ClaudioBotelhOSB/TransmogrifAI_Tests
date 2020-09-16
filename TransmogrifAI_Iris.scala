%classpath add mvn com.salesforce.transmogrifai transmogrifai-core_2.11 0.7.0

%classpath add mvn org.apache.spark spark-mllib_2.11 2.4.5

case class Iris
(
  sepalLength: Double,
  sepalWidth: Double,
  petalLength: Double,
  petalWidth: Double,
  irisClass: String
)

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

val sepalLength = FeatureBuilder.Real[Iris].extract(_.sepalLength.toReal).asPredictor
val sepalWidth = FeatureBuilder.Real[Iris].extract(_.sepalWidth.toReal).asPredictor
val petalLength = FeatureBuilder.Real[Iris].extract(_.petalLength.toReal).asPredictor
val petalWidth = FeatureBuilder.Real[Iris].extract(_.petalWidth.toReal).asPredictor
val irisClass = FeatureBuilder.Text[Iris].extract(_.irisClass.toText).asResponse

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.udf

val conf = new SparkConf().setMaster("local[*]").setAppName("TitanicPrediction")
implicit val spark = SparkSession.builder.config(conf).getOrCreate()

import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.readers.DataReaders
import com.salesforce.op.stages.impl.classification.MultiClassificationModelSelector
import com.salesforce.op.stages.impl.tuning.DataCutter
import org.apache.spark.sql.Encoders

implicit val irisEncoder = Encoders.product[Iris]

val irisReader = DataReaders.Simple.csvCase[Iris]()

val labels = irisClass.indexed()
val features = Seq(sepalLength, sepalWidth, petalLength, petalWidth).transmogrify()


val randomSeed = 42L
val cutter = DataCutter(reserveTestFraction = 0.2, seed = randomSeed)

val prediction = MultiClassificationModelSelector
    .withCrossValidation(splitter = Option(cutter), seed = randomSeed)
    .setInput(labels, features).getOutput()

val evaluator = Evaluators.MultiClassification.f1().setLabelCol(labels).setPredictionCol(prediction)

implicit val spark = SparkSession.builder.config(conf).getOrCreate()
import spark.implicits._ // Needed for Encoders for the Passenger case class
import com.salesforce.op.readers.DataReaders

val trainFilePath = "/home/beakerx/helloworld/src/main/resources/IrisDataset/iris.data"
    // Define a way to read data into our Passenger class from our CSV file
val trainDataReader = DataReaders.Simple.csvCase[Iris](
      path = Option(trainFilePath)
      //key = _.id.toString
    )

val workflow = new OpWorkflow().setResultFeatures(prediction, labels).setReader(trainDataReader)

val fittedWorkflow = workflow.train()
println("Summary:\n" + fittedWorkflow.summaryPretty())

println("Scoring the model:\n=================")
val (dataframe, metrics) = fittedWorkflow.scoreAndEvaluate(evaluator = evaluator)

println("Transformed dataframe columns:\n--------------------------")
dataframe.columns.foreach(println)

println("Metrics:\n------------")
println(metrics)


