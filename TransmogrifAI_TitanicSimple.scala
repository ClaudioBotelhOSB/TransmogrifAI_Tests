%classpath add mvn com.salesforce.transmogrifai transmogrifai-core_2.11 0.7.0

%classpath add mvn org.apache.spark spark-mllib_2.11 2.4.5

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.udf

import com.salesforce.op._
import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.evaluators.Evaluators

val conf = new SparkConf().setMaster("local[*]").setAppName("TitanicPrediction")
implicit val spark = SparkSession.builder.config(conf).getOrCreate()

case class Passenger(
  id: Int,
  survived: Int,
  pClass: Option[Int],
  name: Option[String],
  sex: Option[String],
  age: Option[Double],
  sibSp: Option[Int],
  parCh: Option[Int],
  ticket: Option[String],
  fare: Option[Double],
  cabin: Option[String],
  embarked: Option[String]
)

val survived = FeatureBuilder.RealNN[Passenger].extract(_.survived.toRealNN).asResponse
val pClass = FeatureBuilder.PickList[Passenger].extract(_.pClass.map(_.toString).toPickList).asPredictor
val name = FeatureBuilder.Text[Passenger].extract(_.name.toText).asPredictor
val sex = FeatureBuilder.PickList[Passenger].extract(_.sex.map(_.toString).toPickList).asPredictor
val age = FeatureBuilder.Real[Passenger].extract(_.age.toReal).asPredictor
val sibSp = FeatureBuilder.Integral[Passenger].extract(_.sibSp.toIntegral).asPredictor
val parCh = FeatureBuilder.Integral[Passenger].extract(_.parCh.toIntegral).asPredictor
val ticket = FeatureBuilder.PickList[Passenger].extract(_.ticket.map(_.toString).toPickList).asPredictor
val fare = FeatureBuilder.Real[Passenger].extract(_.fare.toReal).asPredictor
val cabin = FeatureBuilder.PickList[Passenger].extract(_.cabin.map(_.toString).toPickList).asPredictor
val embarked = FeatureBuilder.PickList[Passenger].extract(_.embarked.map(_.toString).toPickList).asPredictor

val familySize = sibSp + parCh + 1
val estimatedCostOfTickets = familySize * fare
val pivotedSex = sex.pivot()
val normedAge = age.fillMissingWithMean().zNormalize()
val ageGroup = age.map[PickList](_.value.map(v => if (v > 18) "adult" else "child").toPickList)


val passengerFeatures = Seq(
      pClass, name, age, sibSp, parCh, ticket,
      cabin, embarked, familySize, estimatedCostOfTickets,
      pivotedSex, ageGroup
    ).transmogrify()

val sanityCheck = true
val finalFeatures = if (sanityCheck) survived.sanityCheck(passengerFeatures) else passengerFeatures

import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelsToTry._

val prediction =
      BinaryClassificationModelSelector.withTrainValidationSplit(
        modelTypesToUse = Seq(OpLogisticRegression)
      ).setInput(survived, finalFeatures).getOutput()


val evaluator = Evaluators.BinaryClassification().setLabelCol(survived).setPredictionCol(prediction)

import spark.implicits._ // Needed for Encoders for the Passenger case class
import com.salesforce.op.readers.DataReaders

val trainFilePath = "/home/beakerx/helloworld/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv"
    // Define a way to read data into our Passenger class from our CSV file
val trainDataReader = DataReaders.Simple.csvCase[Passenger](
      path = Option(trainFilePath),
      key = _.id.toString
    )

val workflow =
      new OpWorkflow()
        .setResultFeatures(survived, prediction)
        .setReader(trainDataReader)

val fittedWorkflow = workflow.train()
println("Summary:\n" + fittedWorkflow.summaryPretty())

println("Scoring the model:\n=================")
val (dataframe, metrics) = fittedWorkflow.scoreAndEvaluate(evaluator = evaluator)

println("Transformed dataframe columns:\n--------------------------")
dataframe.columns.foreach(println)

println("Metrics:\n------------")
println(metrics)


