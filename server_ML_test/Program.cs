using System;
using System.IO;
using Microsoft.Data.Analysis;

class MainStart
{
    static void Main(string[] args)
    {
        // Точка входа в программу
        Microsoft.Data.Analysis.DataFrame df= GetPath();
        Console.WriteLine(df);
    }

    static Microsoft.Data.Analysis.DataFrame GetPath()
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
      
        var dataPath = Path.GetFullPath(@"ts2025.csv");
        
        // Load the data into the data frame
        var df = DataFrame.LoadCsv(dataPath);
        
        return df;
        
    }
}
