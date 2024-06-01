using System;
using System.IO;
using Microsoft.Data.Analysis;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using System.IO;

class MainStart
{
    static void Main(string[] args)
    {
        // Точка входа в программу
        var df = GetPath();
        // Console.WriteLine(df);
        ModelTrain(df);
    }


    static (string, string) GetPath()
    {
        // Путь к каталогу, в который нужно перейти
        string targetDirectory = @"E:\Сommercial projects\C#\server_ML_test\server_ML_test\";

        try
        {
            // Устанавливаем текущий каталог
            Directory.SetCurrentDirectory(targetDirectory);
            Console.WriteLine("Успешно перешли в каталог: " + targetDirectory);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Ошибка при переходе в каталог: " + ex.Message);
        }

        {
        }

        var dataPathTrain = Path.GetFullPath(@"train.csv");
        var dataPathTest = Path.GetFullPath(@"test.csv");

        // Load the data into the data frame
        // var df = DataFrame.LoadCsv(dataPath);

        // return df;

        return (dataPathTrain, dataPathTest);
    }


    static void ModelTrain((string, string) df)
    {
        string path_train = df.Item1;
        string path_test = df.Item2;
        
        
        
        MLContext mlContext = new MLContext();
        
        IDataView data1View =
            mlContext.Data.LoadFromTextFile<ModelInput>(path_train, separatorChar: ',', hasHeader: false);
        IDataView data2View =
            mlContext.Data.LoadFromTextFile<ModelInput>(path_test, separatorChar: ',', hasHeader: false);
        
        
        var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
            outputColumnName: "forecasted_count",
            inputColumnName: "count",
            windowSize: 7,
            seriesLength: 12,
            trainSize: 40,
            horizon: 7,
            confidenceLevel: 0.95f,
            confidenceLowerBoundColumn: "LowerBoundRentals",
            confidenceUpperBoundColumn: "UpperBoundRentals");
        
        SsaForecastingTransformer forecaster = forecastingPipeline.Fit(data1View);

        var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
    }
    
}


public class ModelInput
{
    [LoadColumn(1)] public DateTime action_time { get; set; }
    [LoadColumn(0)] public float count { get; set; }
}

public class TrainData
{
    [LoadColumn(1)] public DateTime action_time { get; set; }

    [LoadColumn(0)] public float count { get; set; }


    // Добавьте другие поля, если они присутствуют в ваших данных
}

public class ModelOutput
{
    public float[] forecasted_count { get; set; }
    public float[] lower_count { get; set; }
    public float[] upper_count { get; set; }
}

