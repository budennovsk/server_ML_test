using System;
using System.IO;
using Microsoft.Data.Analysis;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using System.IO;
using System;
using System.Collections.Generic;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.CompilerServices;


class MainStart
{
    static void Main(string[] args)
    {
        // Точка входа в программу
        var df = GetPath();
        // Console.WriteLine(df);
        var result_tuple = ModelTrain(df);
        // Console.WriteLine(result_tuple);
        var train_data = result_tuple.Item2;
        var forecast_data = result_tuple.Item1;


        PlotTest(train_data, forecast_data);
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

    static void PlotTest(float[] train_data, float[] forecast_data)
    {
        
        int[] numbers_train = Enumerable.Range(1, 50).ToArray();
        int[] numbers_forecast = Enumerable.Range(51, forecast_data.Length).ToArray();
        
        
        Bitmap bmp = new Bitmap(800, 600);
        using (Graphics g = Graphics.FromImage(bmp))
        {
            g.Clear(Color.White);

            // Рисуем обучающие данные (синий)
            DrawGraph(g, numbers_train, train_data, Color.Blue);

            // Рисуем данные прогноза (красный)
            DrawGraph(g, numbers_forecast, forecast_data, Color.Red);
        }

        // Сохраняем график в файл PNG
        bmp.Save("plot.png", System.Drawing.Imaging.ImageFormat.Png);
    }
    static void DrawGraph(Graphics g, int[] xValues, float[] yValues, Color color)
    {
        Pen pen = new Pen(color);
        int width = 800;
        int height = 600;
        int padding = 50;
        float scaleX = (float)(width - 2 * padding) / (xValues.Length - 1);
        float scaleY = (float)(height - 2 * padding) / (MaxValue(yValues) - MinValue(yValues));

        int xOffset = color == Color.Blue ? 0 : 400; // Смещение для красных данных

        for (int i = 0; i < xValues.Length - 1; i++)
        {
            int x1 = (int)(padding + i * scaleX + xOffset);
            int y1 = (int)(height - padding - (yValues[i] - MinValue(yValues)) * scaleY);
            int x2 = (int)(padding + (i + 1) * scaleX + xOffset);
            int y2 = (int)(height - padding - (yValues[i + 1] - MinValue(yValues)) * scaleY);
            g.DrawLine(pen, x1, y1, x2, y2);
        }

        pen.Dispose();
    }

    static float MinValue(float[] array)
    {
        float min = float.MaxValue;
        foreach (float value in array)
        {
            if (value < min)
                min = value;
        }
        return min;
    }
    

    static float MaxValue(float[] array)
    {
        float max = float.MinValue;
        foreach (float value in array)
        {
            if (value > max)
                max = value;
        }
        return max;
    }





    static (float[],float[]) ModelTrain((string, string) df)
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
            windowSize: 8,
            seriesLength: 30,
            trainSize: 50,
            horizon: 50,
            confidenceLevel: 0.95f,
            confidenceLowerBoundColumn: "lower_count",
            confidenceUpperBoundColumn: "upper_count");

        SsaForecastingTransformer forecaster = forecastingPipeline.Fit(data1View);

        var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
        ModelOutput forecast = forecastEngine.Predict();
        Console.WriteLine($"Forecast for the next 7 time points: {string.Join(", ", forecast.forecasted_count)}");
        // var forecast_string = string.Join(", ", forecast.forecasted_count);
        var df_train = DataFrame.LoadCsv(path_train);

        

        // Считываем все строки из файла
        string[] lines = File.ReadAllLines(path_train);
        int ii = 0;
        // Создаем массивы для хранения данных
        float[] floatArray = new float[lines.Length-1];
        Console.WriteLine(floatArray.Length);
        // Перебираем строки CSV-файла и разбираем значения
        
        for (int i = 1; i < lines.Length-1; i++)
        {   
            string modifiedString_1 = lines[i].Replace(',', ';');
            string modifiedString_2 = modifiedString_1.Replace('.', ',');
        
            string[] values = modifiedString_2.Split(';');
          

            // Парсим значения в типы float и int
            float floatValue = float.Parse(values[0]); // Первый столбец в C
            // Console.WriteLine(floatValue);
            // Сохраняем значения в массивы
            floatArray[i] = floatValue;
            
        }

        // foreach (var item in floatArray)
        // {
        //     Console.WriteLine(item);
        // }
        //

        return (forecast.forecasted_count,floatArray);

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

