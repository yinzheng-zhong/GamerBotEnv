from src.Model.data_pipeline import DataPipeline

if __name__ == "__main__":
    dp = DataPipeline()
    while True:
        #dp.print_text()
        dp.make_batch()
        #print("\n")