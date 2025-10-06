import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomBinningStrategy:
    def __init__(self, bin_definitions):
        self.bin_definitions = bin_definitions
        logger.info(f"CustomBinningStrategy initialized with bins: {list(bin_definitions.keys())}")

    def bin_feature(self, df, column):
        df_copy = df.copy()
        bin_column = f'{column}Bins'
        
        def assign_bin(value):
            for bin_label, bin_range in self.bin_definitions.items():
                if len(bin_range) == 2:
                    if bin_range[0] <= value <= bin_range[1]:
                        return bin_label
                elif len(bin_range) == 1:
                    if value >= bin_range[0]:
                        return bin_label
            return None
        
        if column in df_copy.columns:
            df_copy[bin_column] = df_copy[column].apply(assign_bin)
            df_copy = df_copy.drop(columns=[column])
            logger.info(f"Binned column {column} into {bin_column}")
        
        return df_copy