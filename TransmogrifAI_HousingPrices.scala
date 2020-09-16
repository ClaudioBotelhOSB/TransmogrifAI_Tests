%classpath add mvn com.salesforce.transmogrifai transmogrifai-core_2.11 0.7.0

%classpath add mvn org.apache.spark spark-mllib_2.11 2.4.5

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.udf

import com.salesforce.op._
import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.evaluators.Evaluators

import com.salesforce.op.OpWorkflow
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.readers.DataReaders

val conf = new SparkConf().setMaster("local[*]").setAppName("HousingPricesPrediction")
implicit val spark = SparkSession.builder.config(conf).getOrCreate()

case class HousingPrices(
  lotFrontage: Double,
  area: Integer,
  lotShape: String,
  yrSold : Integer,
  saleType: String,
  saleCondition: String,
  salePrice: Double)

import org.apache.spark.sql.{Encoders}
implicit val srEncoder = Encoders.product[HousingPrices]

val lotFrontage = FeatureBuilder.Real[HousingPrices].extract(_.lotFrontage.toReal).asPredictor
val area = FeatureBuilder.Integral[HousingPrices].extract(_.area.toIntegral).asPredictor

val lotShape = FeatureBuilder.Integral[HousingPrices].extract(_.lotShape match {
    case "IR1" => 1.toIntegral
    case _ => 0.toIntegral
}).asPredictor

val yrSold = FeatureBuilder.Integral[HousingPrices].extract(_.yrSold.toIntegral).asPredictor

val saleType = FeatureBuilder.Text[HousingPrices].extract(_.saleType.toText).asPredictor.indexed()

val saleCondition = FeatureBuilder.Text[HousingPrices]
  .extract(_.saleCondition.toText).asPredictor.indexed()

val salePrice = FeatureBuilder.RealNN[HousingPrices].extract(_.salePrice.toRealNN).asResponse

 val trainFilePath = "/home/beakerx/helloworld/src/main/resources/HousingPricesDataset/train_lf_la_ls_ys_st_sc.csv"

val trainDataReader = DataReaders.Simple.csvCase[HousingPrices](
      path = Option(trainFilePath)
    )

import com.salesforce.op.stages.impl.tuning.{DataCutter, DataSplitter}
val features = Seq(lotFrontage,area,lotShape, yrSold, saleType, saleCondition).transmogrify()
val randomSeed = 42L
val splitter = DataSplitter(seed = randomSeed)

import com.salesforce.op.stages.impl.regression.RegressionModelSelector
import com.salesforce.op.stages.impl.regression.RegressionModelsToTry.{OpGBTRegressor, OpRandomForestRegressor}

val prediction1 = RegressionModelSelector
      .withCrossValidation(
        dataSplitter = Some(splitter), seed = randomSeed,
        modelTypesToUse = Seq(OpGBTRegressor, OpRandomForestRegressor)
      ).setInput(salePrice,features).getOutput()

val evaluator = Evaluators.Regression().setLabelCol(salePrice).setPredictionCol(prediction1)

val workflow = new OpWorkflow().setResultFeatures(prediction1, salePrice).setReader(trainDataReader)
val workflowModel = workflow.train()

val (scores, metrics) = workflowModel.scoreAndEvaluate(evaluator)
scores.show(false)

metrics.toString()

println("Metrics:\n------------")
println(metrics)


