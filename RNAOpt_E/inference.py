from utils import RNADataset_finetune, RNATokenizer, load_config
from model import RNA_MaskedLM_finetune
from torch.utils.data import Dataset, DataLoader
import torch    
from tqdm import tqdm
import pandas as pd


if __name__ == "__main__":
    # Load configuration
    config = load_config('/home/reagan/Projects/RNA_optimization/model/models_put_on_github/model_modules/configs/inference_config.yaml')

    # Initialize tokenizer
    tokenizer = RNATokenizer()

    # Create dataset and dataloader
    dataset = RNADataset_finetune(config['data_file'], tokenizer, config['max_len'], specie="Human")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=48)

    # Initialize model
    model = RNA_MaskedLM_finetune(config=config)

    # Load pretrained weights
    checkpoint = torch.load(config['finetuned_model_path'])
    full_state_dict = checkpoint['state_dict']

    # torch.save(modified_state_dict, "14M_stage2_finetuned_on_human_mouse_new.ckpt")

    model.load_state_dict(full_state_dict, strict=True)

    model.eval().to("cuda:0")
    all_prediction_list = []
    all_label_list = []
    for batch in tqdm(dataloader):
        
        with torch.no_grad():
            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")
            
            value, _, _ = model(input_ids, attention_mask)
            
            #print(input_ids)
       
            all_prediction_list.extend([ele[0] for ele in value.tolist()])
            all_label_list.extend(labels.tolist())

    import scipy.stats as stats

    # Example data
    A = all_label_list  # Real half-life values
    B = all_prediction_list  # Predicted half-life values
    #Encodon [2.942596197128296, 2.6468183994293213, 2.8545191287994385, 2.631762742996216, 2.605633020401001, 0.661151647567749, 0.43522781133651733, 0.04598169028759003, 0.23917852342128754, 0.40615302324295044, 3.10764479637146, 3.6161749362945557, 3.9937222003936768, 3.688199758529663, 3.8492140769958496, 1.2188392877578735, 2.844789743423462, 4.260601997375488, 0.6016179323196411, 1.8766096830368042, 4.554112434387207, 5.236441612243652, 4.96204948425293, 3.0665268898010254, 2.772230386734009, -1.2418304681777954, 0.014414206147193909, -0.5937954783439636, -0.839163064956665, -0.49597978591918945]

    # df = pd.DataFrame({"predictions": B})
    # df.to_csv("/home/reagan/Projects/RNA_optimization/model/models_put_on_github/sanity_check/prediction_AAUAAA/noAAUAAA_seqs_predictions.csv", index=False)
    # print(B)
    # our model [1.9951919317245483, 0.7416497468948364, 1.366500973701477, 1.2175582647323608, 1.8903175592422485, 0.722223162651062, -2.3981146812438965, -0.38228389620780945, 0.1628650426864624, -0.6423249244689941, 1.9829988479614258, 2.8806052207946777, 1.974485158920288, 0.3264981508255005, 0.557052493095398, 0.47718000411987305, -0.16123710572719574, -0.4325971007347107, -0.9049665331840515, 0.8464956879615784, 1.93472158908844, 2.9458329677581787, 1.2437292337417603, -0.30508744716644287, 0.6009541153907776, -1.1961400508880615, -1.2191557884216309, -2.4556078910827637, -2.727421760559082, -1.8464359045028687]  # Predicted half-life values

    # Calculate Spearman's rank correlation coefficient
    spearman_corr, p_value = stats.spearmanr(A, B)
    pearson_corr, p_value2 = stats.pearsonr(A, B)

    # print(B)

    print(f"Pearson's Correlation: {pearson_corr:.4f}")
    print(f"P-value: {p_value2:.4f}")

    print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")
    print(f"P-value: {p_value:.4f}")
        