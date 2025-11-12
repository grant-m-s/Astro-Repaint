# Astro-Repaint

This code accomponies the paper:

**Active galactic nuclei identification using diffusion-based inpainting of Euclid VIS images**

*Euclid Collaboration: Stevens et al. (2025)*

[Arxiv: 2503.15321](https://arxiv.org/abs/2503.15321) | [Euclid Q1 A\&A Special Issue](https://www.aanda.org/component/toc/?task=topic&id=2247) | [Euclid Q1 Data Release](https://www.euclid-ec.org/science/q1/)

<img src="assets/542_0.gif" alt="drawing" width="150"/> <img src="assets/563_0.gif" alt="drawing" width="150"/> <img src="assets/628_0.gif" alt="drawing" width="150"/> <img src="assets/743_0.gif" alt="drawing" width="150"/>

---

Large parts of this codebase are the result of combining and streamlining the [Guided Diffusion]() and  [Repaint](https://github.com/andreas128/RePaint) repositories and so credit must go to the respective developers for most of the training and inpainting pipeline.

---

## 1. Creating the Data

Cutouts are stored as NumPy arrays (.npy), each containing -bs sources per file.  
Each array stores the raw VIS pixel values for its cutouts.

*Euclid Q1 VIS tiles are available to download at: https://eas.esac.esa.int/sas/ .The tiles will be saved in the form `EUC_MER_BGSUB-MOSAIC-VIS_TILE#########- ... .fits`, where `#########` will be replaced with the respective Tile ID.*

Given a FITS table containing object ID, RA, DEC, and Tile ID values, batches of .npy files can be created with:
````bash
python create_batches.py -c subset_id_tiles.fits -bs 1024 -p 32 -td "tiles" -bd "data"
````

This produces:

- .npy batch files containing the image cutouts  
- batch_sources_full.json — maps batch filenames to source IDs  
- missing_tiles.txt — lists unavailable or missing tiles  
- image_errors.csv — logs failed or corrupted cutouts  

To process a single tile only, add the -t argument:

````bash
python create_batches.py -t 102157953 -c subset_id_tiles.fits -bs 1024 -p 32 -td "tiles" -bd "data"
````

⚠️ **Note**: If a tile contains artefacts (e.g. solar interference), all cutouts in that batch will be affected. Single-tile batches are recommended for inference or testing only.

The input parameters for the create_batches script are the following:

  - **c**: (*catalogue*) The fits catalogue with all required sources (required).
  - **bs**: (*batchsize*) The number of images within each created .npy file.
  - **p**: (*processses*) The number of processes used to create the .npy files.
  - **td**: (*tile_dir*) The directory where all tiles are stored.
  - **bd**: (*batch_dir*) The directory where all .npy files will be saved.
  - **t**: (*tile*) If t is specified, only sources from that tile will be saved to .npy files.

---

## 2. Inpainting

![image](assets/repaint_pipeline_v2.svg)


The specific model used in the paper can be downloaded here:

Model weights (Google Drive): https://drive.google.com/file/d/1q6GFYnLUOyUPagZTozfX-zOk44a5vlje/view?usp=sharing

Save the downloaded file model_best.pt into: `training_tmp/`.

### Interactive Dashboard

You can perform interactive, one-click inpainting using our dashboard:

````bash
panel serve example.py
````

Then open your browser to: http://localhost:5006/example .

Each completed batch automatically saves a results file in: `repaint_data/`

This enables automatic continuation if the process is stopped. Changing dashboard parameters (e.g. number of iterations, inpaint mask size) creates distinct result files.

The dashboard operates on one .npy file at a time for easy inspection.  

### Inpainting CLI Script

To inpaint all data non-interactively via the command line, use:

````bash
python repaint.py --conf_path confs/galaxy.yml -t 60 -js 3 -jl 1 -is 5 -bs 16
````

This command also checkpoints progress after each iteration, allowing pause-and-resume inference.

The input parameters for the inpainting scripts are the following:

  - **conf_path**: Path to configuration file. 
  - **is**: (*inpaint_size*) The size of the square mask used for inpainting.
  - **t**: The number of inference timesteps. *Larger numbers will produce more resiliant outputs but will take longer per batch.*
  - **js**: (*jump_n_samples*) The number of resamples made in the repaint process (see repaint paper for details). *Larger numbers will produce more resiliant outputs but will take longer per batch.*
  - **jl**: (*jump_length*) The number of steps jumped before resampling. *Smaller numbers will produce more resiliant outputs but will take longer per batch.*
  - **bs**: (*batch_size*) The number of images to inpaint at once in an iteration. 
  
  ⚠️ **Note**: Memory is the main bottleneck for inference and so only increase the batchsize if you have sufficient GPU Memory.

![image](assets/2643931528666914057_5.svg)
![image](assets/2733948397649875934_5.svg)
![image](assets/-621286250473882410_5.svg)

---

## 3. Training the Model

To train a new model using the same parameters as in the paper:

````bash
export PYTHONPATH=.; python scripts/image_train.py --num_channels 32 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine --lr 1e-4 --batch_size 1 --image_size 64 --loss_function "1_over_pixel"
````

After training, update the model path for inpainting in `confs/galaxy.yml`, setting the `model_path` field to point to your newly trained model.


The specific input parameters we have added to the training script are the following:

  - **batch_size**: This batchsize corresponds to the number of .npy files being loaded in at one time for training. 
  - **image_size**: This is size of the cutouts inputted into the model. If this is smaller than the image sizes in the .npy files, they will be cropped from the centre of the cutouts.
  - **loss_function**: The loss function used during training.

  ⚠️ **Note**: All other model parameters are used specifically as shown in the OpenAI [Guided Diffusion](https://github.com/openai/guided-diffusion) and [Improved Diffusion](https://github.com/openai/improved-diffusion) repositories. The reader is encouraged to look at their documentation for further details on altering these parameters.


---

## 4. Notes

- The diffusion and inpainting pipelines correspond to Sections 2 - 3 of the Euclid paper.  
- The "1_over_pixel" loss corresponds to the normalised MSE + VLB hybrid described in Eq. (9) of the paper.  
- Data creation uses raw VIS pixel values (no rescaling), as detailed in Section 3.2 (Reconstruction Rescaling) of the paper.  
- Inpainting is implemented via the Repaint algorithm (Lugmayr et al. 2022), ensuring consistent conditioning between masked and unmasked pixels.

---

## Reference

If you use this code or model, please cite:

```
Euclid Collaboration: G. Stevens et al. (2025),  "Active galactic nuclei identification using diffusion-based inpainting of Euclid VIS images", Astronomy & Astrophysics, 2025.
```

or use the following bib entry:


````latex
@article{ Stevens2025Euclid,
	author = {{Euclid Collaboration: Stevens, G.} and {Fotopoulou, S.} and {Bremer, M.N.} and {Matamoro Zatarain, T.} and {Jahnke, K.} and {Margalef-Bentabol, B.} and {Huertas-Company, M.} and {Smith, M.J.} and {Walmsley, M.} and {Salvato, M.} and {Mezcua, M.} and {Paulino-Afonso, A.} and {Siudek, M.} and {Talia, M.} and {Ricci, F.} and {Roster, W.} and {Aghanim, N.} and {Altieri, B.} and {Andreon, S.} and {Aussel, H.} and {Baccigalupi, C.} and {Baldi, M.} and {Bardelli, S.} and {Battaglia, P.} and {Biviano, A.} and {Bonchi, A.} and {Branchini, E.} and {Brescia, M.} and {Brinchmann, J.} and {Camera, S.} and {Cañas-Herrera, G.} and {Capobianco, V.} and {Carbone, C.} and {Carretero, J.} and {Castellano, M.} and {Castignani, G.} and {Cavuoti, S.} and {Chambers, K.C.} and {Cimatti, A.} and {Colodro-Conde, C.} and {Congedo, G.} and {Conselice, C.J.} and {Conversi, L.} and {Copin, Y.} and {Costille, A.} and {Courbin, F.} and {Courtois, H.M.} and {Cropper, M.} and {Da Silva, A.} and {Degaudenzi, H.} and {De Lucia, G.} and {Dolding, C.} and {Dole, H.} and {Douspis, M.} and {Dubath, F.} and {Dupac, X.} and {Dusini, S.} and {Escoffier, S.} and {Farina, M.} and {Ferriol, S.} and {George, K.} and {Giocoli, C.} and {Granett, B.R.} and {Grazian, A.} and {Grupp, F.} and {Haugan, S.V.H.} and {Hook, I.M.} and {Hormuth, F.} and {Hornstrup, A.} and {Hudelot, P.} and {Jhabvala, M.} and {Keihänen, E.} and {Kermiche, S.} and {Kiessling, A.} and {Kilbinger, M.} and {Kubik, B.} and {Kümmel, M.} and {Kurki-Suonio, H.} and {Le Boulc'h, Q.} and {Le Brun, A.M.C.} and {Le Mignant, D.} and {Lilje, P.B.} and {Lindholm, V.} and {Lloro, I.} and {Mainetti, G.} and {Maino, D.} and {Maiorano, E.} and {Marggraf, O.} and {Martinelli, M.} and {Martinet, N.} and {Marulli, F.} and {Massey, R.} and {Maurogordato, S.} and {McCracken, H.J.} and {Medinaceli, E.} and {Mei, S.} and {Melchior, M.} and {Meneghetti, M.} and {Merlin, E.} and {Meylan, G.} and {Mora, A.} and {Moresco, M.} and {Moscardini, L.} and {Nakajima, R.} and {Neissner, C.} and {Niemi, S.-M.} and {Padilla, C.} and {Paltani, S.} and {Pasian, F.} and {Pedersen, K.} and {Percival, W.J.} and {Pettorino, V.} and {Polenta, G.} and {Poncet, M.} and {Popa, L.A.} and {Pozzetti, L.} and {Raison, F.} and {Rebolo, R.} and {Renzi, A.} and {Rhodes, J.} and {Riccio, G.} and {Romelli, E.} and {Roncarelli, M.} and {Saglia, R.} and {Sánchez, A.G.} and {Sapone, D.} and {Schewtschenko, J.A.} and {Schirmer, M.} and {Schneider, P.} and {Schrabback, T.} and {Secroun, A.} and {Serrano, S.} and {Simon, P.} and {Sirignano, C.} and {Sirri, G.} and {Skottfelt, J.} and {Stanco, L.} and {Steinwagner, J.} and {Tallada-Crespí, P.} and {Taylor, A.N.} and {Tereno, I.} and {Toft, S.} and {Toledo-Moreo, R.} and {Torradeflot, F.} and {Tutusaus, I.} and {Valenziano, L.} and {Valiviita, J.} and {Vassallo, T.} and {Verdoes Kleijn, G.} and {Veropalumbo, A.} and {Wang, Y.} and {Weller, J.} and {Zacchei, A.} and {Zamorani, G.} and {Zerbi, F.M.} and {Zinchenko, I.A.} and {Zucca, E.} and {Allevato, V.} and {Ballardini, M.} and {Bolzonella, M.} and {Bozzo, E.} and {Burigana, C.} and {Cabanac, R.} and {Cappi, A.} and {Escartin Vigo, J.A.} and {Gabarra, L.} and {Hartley, W.G.} and {Martín-Fleitas, J.} and {Matthew, S.} and {Metcalf, R.B.} and {Pezzotta, A.} and {Pöntinen, M.} and {Risso, I.} and {Scottez, V.} and {Sereno, M.} and {Tenti, M.} and {Wiesmann, M.} and {Akrami, Y.} and {Alvi, S.} and {Andika, I.T.} and {Anselmi, S.} and {Archidiacono, M.} and {Atrio-Barandela, F.} and {Bertacca, D.} and {Bethermin, M.} and {Bisigello, L.} and {Blanchard, A.} and {Blot, L.} and {Borgani, S.} and {Brown, M.L.} and {Bruton, S.} and {Calabro, A.} and {Caro, F.} and {Castro, T.} and {Cogato, F.} and {Davini, S.} and {Desprez, G.} and {Díaz-Sánchez, A.} and {Diaz, J.} and {Di Domizio, S.} and {Diego, J.M.} and {Duc, P.-A.} and {Enia, A.} and {Fang, Y.} and {Ferrari, A.G.} and {Finoguenov, A.} and {Fontana, A.} and {Franco, A.} and {García-Bellido, J.} and {Gasparetto, T.} and {Gautard, V.} and {Gaztanaga, E.} and {Giacomini, F.} and {Gianotti, F.} and {Guidi, M.} and {Gutierrez, C.M.} and {Hall, A.} and {Hemmati, S.} and {Hildebrandt, H.} and {Hjorth, J.} and {Kajava, J.E.} and {Kang, Y.} and {Kansal, V.} and {Karagiannis, D.} and {Kirkpatrick, C.} and {Kruk, S.} and {Legrand, L.} and {Lembo, M.} and {Lepori, F.} and {Leroy, G.} and {Lesgourgues, J.} and {Leuzzi, L.} and {Liaudat, T.I.} and {Macias-Perez, J.} and {Magliocchetti, M.} and {Mannucci, F.} and {Maoli, R.} and {Martins, C.J.A.P.} and {Maurin, L.} and {Miluzio, M.} and {Monaco, P.} and {Morgante, G.} and {Naidoo, K.} and {Navarro-Alsina, A.} and {Passalacqua, F.} and {Paterson, K.} and {Patrizii, L.} and {Pisani, A.} and {Potter, D.} and {Quai, S.} and {Radovich, M.} and {Rocci, P.-F.} and {Rodighiero, G.} and {Sacquegna, S.} and {Sahlén, M.} and {Sanders, D.B.} and {Sarpa, E.} and {Schneider, A.} and {Schultheis, M.} and {Sciotti, D.} and {Sellentin, E.} and {Shankar, F.} and {Smith, L.C.} and {Tanidis, K.} and {Testera, G.} and {Teyssier, R.} and {Tosi, S.} and {Troja, A.} and {Tucci, M.} and {Valieri, C.} and {Vergani, D.} and {Verza, G.} and {Walton, N.A.}},
	title = {Euclid Quick Data Release (Q1). Active galactic nuclei identification using diffusion-based inpainting of Euclid VIS images},
	DOI= "10.1051/0004-6361/202554612",
	url= "https://doi.org/10.1051/0004-6361/202554612",
	journal = {A&A},
	year = 2025,
}

````


---
