import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.metrics import structural_similarity as ssim
import math
import concurrent.futures
import time
start = time.time()


def GA(tile: np.ndarray):
    # Parameters for the Genetic Algorithm
    population_size = 10
    generations = 10
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Initialize population
    clip_limit_values = np.arange(0.01, 0.51, 0.03)

    # Initialize population with random selection from clip_limit_values
    population = random.sample(list(clip_limit_values), population_size)
    
    # Genetic Algorithm
    for generation in range(generations):
        # Evaluate fitness of each individual
        fitness_scores = [(clip_limit, fitness(tile, clip_limit)) for clip_limit in population]
        
        # Select the best individuals
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        population = [clip_limit for clip_limit, score in fitness_scores[:population_size // 2]]
        
        # Create next generation
        next_generation = []
        while len(next_generation) < population_size:
            # Crossover
            if random.random() < crossover_rate:
                parent1, parent2 = random.sample(population, 2)
                child = (parent1 + parent2) / 2
            else:
                child = random.choice(population)
                
            # Mutation
            if random.random() < mutation_rate:
                child += random.uniform(-0.01, 0.01)
                child = max(0.01, min(0.5, child))  # Keep within bounds
            
            next_generation.append(child)
        
        population = next_generation
    
    # Return the best clip limit from the final population
    best_clip_limit = max(population, key=lambda clip_limit: fitness(tile, clip_limit))
    print(f'best clip limit = {best_clip_limit:.2f}, fitnesss score = {fitness(tile, best_clip_limit)*10}')
    return best_clip_limit

def De(temp):
    temp_hist, _ = np.histogram(temp.flatten(),65535,[0,65535])
    
    temp_hist = temp_hist / temp_hist.sum()

    # Step 4: Calculate entropy using the formula
    temp_hist = temp_hist[temp_hist > 0]
    entropy = -np.sum(temp_hist * np.log2(temp_hist))
    
    return entropy

def fitness(tile, clip_limit):
    temp, _ = apply_clahe_to_tile_cl(tile, clip_limit)
    de = De(temp)
    ssim_value = ssim(temp, tile, data_range=tile.max() - tile.min())
    return 0.6*(de-1)/3 + 0.2* ssim_value

def calculate_clip_limit(hist: np.ndarray, clip_pixels: np.float64) -> tuple:
    clipped = np.copy(hist)
    while(clip_pixels > 0 ):
        sorted_indices = np.argsort(hist)[::-1]
        max_value = np.max(clipped)
        max_bin_indices = np.where(clipped == max_value)[0]
        second_max_bin_index = sorted_indices[len(max_bin_indices)]
        second_max_value = hist[second_max_bin_index]
        gap = max_value - second_max_value
        # Reduce the values of bins with the highest value.
        for i in max_bin_indices:
            clipped[i] -= gap
            clip_pixels -= gap

    clip_limit = clipped.max() # The final clip limit.

    return clip_limit, clipped

def clip_histogram(hist: np.ndarray, clip_limit: float) -> np.ndarray:
    clip_pixels=hist.sum()*clip_limit
    clip_limit , clipped = calculate_clip_limit(hist,clip_pixels)
    excess = np.maximum(0, hist - clip_limit)
    e = np.sum(excess)
    hist_clipped =  np.minimum(hist, clip_limit)
            
    # Compute the probability density function (PDF) for values below the threshold
    
    C = hist[hist < clip_limit]
    if C.sum() ==0:
         return hist
    pk = hist / C.sum()
     # Create the adjusted histogram
    adjusted_histogram = np.copy(clipped)
    mask = adjusted_histogram < clip_limit
    a = clip_limit+1
    b = 0
    while a > clip_limit:
        adjusted_histogram =np.where(mask,hist_clipped+ pk * e,hist_clipped)
        prev_e = e
        e = np.sum(np.maximum(0, adjusted_histogram - clip_limit))
        hist_clipped =  np.minimum(adjusted_histogram, clip_limit)
        a = round(adjusted_histogram.max())
        if  b ==10:
            break
        elif e == prev_e and b <10:
            b+=1

    return adjusted_histogram
    
def apply_clahe_to_tile(tile: np.ndarray ) -> tuple:
    hist, bins = np.histogram(tile.flatten(), 65535, [0, 65535])
    clip_limit = GA(tile)
    new_hist = clip_histogram(hist , clip_limit)
    cdf = new_hist.cumsum()
    cdf_normalized =  np.floor((cdf - cdf.min()) / (cdf.max() - cdf.min()) *255)
    
    #print(f'cdf max = {cdf_normalized.max()},cdf min={cdf_normalized.min()}')
    tile_equalized  = cdf_normalized[tile].astype("uint8")

    return (tile_equalized, cdf_normalized)

def apply_clahe_to_tile_cl(tile: np.ndarray, clip_limit            ) -> tuple:
    hist, bins = np.histogram(tile.flatten(), 65535, [0, 65535])
    new_hist = clip_histogram(hist , clip_limit)
    cdf = new_hist.cumsum()
    cdf_normalized =  np.floor((cdf - cdf.min()) / (cdf.max() - cdf.min()) *255)
    
    #print(f'cdf max = {cdf_normalized.max()},cdf min={cdf_normalized.min()}')
    tile_equalized  = cdf_normalized[tile].astype("uint8")

    return (tile_equalized, cdf_normalized)

def clahe_apply(image: np.ndarray, tile_grid_size: tuple) -> tuple:
    h, w = image.shape
    tile_h, tile_w = h // tile_grid_size[0], w // tile_grid_size[1]
    clahe_image = np.zeros_like(image)
    tile_cdf_dict={}
    unprocessed_tiles = []
    #process each tile
    for i in range(tile_grid_size[0]):
        for j in range(tile_grid_size[1]):
            x1 , y1 = i * tile_h , j * tile_w
            x2 , y2 = min(x1 + tile_h,h), min(y1 + tile_w,w)
            tile = image[x1:x2 , y1:y2] #ortak okuma
            unprocessed_tiles.append(tile)
            
    processed_tiles = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_tiles_ = executor.map(apply_clahe_to_tile, unprocessed_tiles)
        for tile in processed_tiles_:
            processed_tiles.append(tile)
    
    counter = 0
    for i in range(tile_grid_size[0]):
        for j in range(tile_grid_size[1]):
            x1, y1 = i * tile_h , j * tile_w
            x2, y2 = min(x1 + tile_h, h), min(y1 + tile_w, w)

            clahe_tile, cdf_to_keep = processed_tiles[counter][0], processed_tiles[counter][1]
            counter += 1
            clahe_image[x1:x2 , y1:y2] = clahe_tile ## Shared Write
            tile_cdf_dict[(i, j)] = cdf_to_keep ## Shared Write
            
            
    output_image = np.zeros_like(image)
   
    for i in range(h) :
        for j in range(w) :
            
            tile_i = i // tile_h #1
            tile_j = j // tile_w #1
            
            tile_i *=  tile_h 
            tile_j *=  tile_w #top left 
            origin_value = image[i,j]
            
            #print(f'cdf max = {origin_cdf.max()},cdf min={origin_cdf.min()}')
            #print(F'org val = {origin_value} ,clahe value ={clahe_image[i,j]}, image={image[i,j]}')
            middle_i = tile_i + tile_h//2
            middle_w = tile_j + tile_w // 2
            #computing new tile
            a = 0.0
            
            if middle_i <= i and middle_w <= j:
                if  middle_i+tile_h < h and middle_w+tile_w < w :
                    a=2.0
                    min_x ,max_x ,min_y ,max_y= middle_i,middle_i+tile_h,middle_w,middle_w+tile_w
                    distA,distB,distC,distD =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif  middle_i+tile_h >= h and middle_w+tile_w < w :
                    a =1.0
                    min_x ,max_x ,min_y ,max_y= middle_i,middle_i,middle_w,middle_w+tile_w
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif  middle_i+tile_h < h and middle_w+tile_w >= w :
                    a=1.1
                    min_x ,max_x ,min_y ,max_y= middle_i,middle_i+tile_h,middle_w,middle_w
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif  middle_i+tile_h >= h and middle_w+tile_w >= w :
                    output_image[i,j] = clahe_image[i,j]
                    continue
                #donedone
                    
            elif middle_i <= i and middle_w >= j:
                    if middle_i+tile_h < h and middle_w-tile_w >=0 :
                        a=2.0
                        min_x ,max_x ,min_y ,max_y=middle_i,middle_i+tile_h,middle_w-tile_w,middle_w
                        distA,distB,distC,distD =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                    elif  middle_i+tile_h >= h and middle_w-tile_w >=0 :
                        a=1.0
                        min_x ,max_x ,min_y ,max_y=middle_i,middle_i,middle_w-tile_w,middle_w
                        distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                    elif  middle_i+tile_h < h and middle_w-tile_w <0 :
                        a=1.1
                        min_x ,max_x ,min_y ,max_y=middle_i,middle_i+tile_h,middle_w,middle_w
                        distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                    elif middle_i+tile_h >= h and middle_w-tile_w < 0 :
                        output_image[i,j] = clahe_image[i,j]
                        continue
                    #Done
            elif(middle_i >= i and middle_w <= j):
                if middle_i-tile_h>=0 and middle_w+tile_w<w-1 :
                    a=2.0
                    min_x ,max_x ,min_y , max_y =max(middle_i-tile_h,0),middle_i,middle_w,min(middle_w+tile_w,w-1)
                    distA,distB,distC,distD =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif middle_i-tile_h<0 and middle_w+tile_w<w :
                    a = 1.0
                    min_x ,max_x ,min_y , max_y =middle_i,middle_i,middle_w,middle_w+tile_w
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif middle_i-tile_h>=0 and middle_w+tile_w>=w :
                    min_x, max_x, min_y, max_y = middle_i-tile_h, middle_i, middle_w, middle_w
                    a =1.1
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif middle_i-tile_h<0 and middle_w+tile_w>=w:
                    output_image[i,j] = clahe_image[i,j]
                    continue
                #done
            elif middle_i > i and middle_w > j:
                if middle_i-tile_h>=0 and middle_w-tile_w>=0:
                    a=2.0
                    min_x ,max_x ,min_y ,max_y=middle_i-tile_h,middle_i,middle_w-tile_w,middle_w
                    distA,distB,distC,distD =dist(i,j,min_x ,max_x ,min_y , max_y,a)                    
                elif middle_i-tile_h<0 and middle_w-tile_w>=0:
                    a  =1.0
                    min_x ,max_x ,min_y ,max_y=middle_i,middle_i,middle_w-tile_w,middle_w
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif middle_i-tile_h>=0 and middle_w-tile_w<0:
                    a = 1.1
                    min_x ,max_x ,min_y ,max_y=middle_i-tile_h,middle_i,middle_w,middle_w
                    distA,distB =dist(i,j,min_x ,max_x ,min_y , max_y,a)
                elif middle_i-tile_h<0 and middle_w-tile_w<0:
                    output_image[i,j] = clahe_image[i,j]
                    continue
                #done
            #values and new pixel values
            if int(a)== 2: 
                valueA, valueB, valueC, valueD = calc_value(origin_value,tile_h,tile_w,min_x ,max_x ,min_y , max_y,tile_cdf_dict,a)
                F, E = (valueA*distB[1] +valueB*distA[1])/tile_w,  (valueC*distD[1] +valueD*distC[1])/tile_w
                new_pixel_value = (F*distC[0] + E*distA[0])/tile_h
                output_image[i,j]= new_pixel_value
                
            elif int(a) == 1:
                valueA, valueB = calc_value(origin_value,tile_h,tile_w,min_x ,max_x ,min_y , max_y,tile_cdf_dict,a)
                new_pixel_value = (valueA*distB + valueB * distA)/(distB +distA)
                if new_pixel_value <0:
                    print(new_pixel_value)
                output_image[i,j]= new_pixel_value
                
            else:
                continue
    #print(clahe_image.shape)
    return (clahe_image, output_image)

def calc_value(origin_value: np.uint8, tile_h: int, tile_w: int, min_x: int ,max_x: int ,min_y: int, max_y: int, tile_cdf_dict: dict, a: float) -> tuple:
    if a ==2.0:
        #A
        tileA_h,tileA_w = min_x // tile_h,min_y // tile_w
        temp_cdf =tile_cdf_dict[(tileA_h,tileA_w)]
        valueA = temp_cdf[origin_value]
        #B
        tileB_h,tileB_w = min_x // tile_h,max_y // tile_w
        temp_cdf =tile_cdf_dict[(tileB_h,tileB_w)]
        valueB = temp_cdf[origin_value]
        #C
        tileC_h,tileC_w = max_x // tile_h,min_y // tile_w
        temp_cdf =tile_cdf_dict[(tileC_h,tileC_w)]
        valueC = temp_cdf[origin_value]
        #D
        tileD_h,tileD_w = max_x // tile_h,max_y // tile_w
        temp_cdf =tile_cdf_dict[(tileD_h,tileD_w)]
        valueD = temp_cdf[origin_value]
        return valueA, valueB, valueC, valueD
    
    elif int(a) ==1:
        tileA_h,tileA_w = min_x // tile_h,min_y // tile_w
        temp_cdf =tile_cdf_dict[(tileA_h,tileA_w)]
        valueA = temp_cdf[origin_value]
    #B
        tileB_h,tileB_w = max_x // tile_h,max_y // tile_w
        temp_cdf =tile_cdf_dict[(tileB_h,tileB_w)]
        valueB = temp_cdf[origin_value]

        return valueA, valueB
    
def dist(i: int, j: int, min_x: int, max_x: int , min_y: int, max_y, a: float) -> tuple:
    li = [i-min_x,i-max_x,j -min_y,j-max_y]
    if a == 2.0 and 0 not in li:
        distA =  abs(i-min_x) , abs(j -min_y)                  
        distB =  abs(i-min_x)  ,  abs(j-max_y)                
        distC =  abs(i-max_x),  abs(j -min_y)                   
        distD =   abs(i-max_x)  , abs(j-max_y)
        return distA, distB, distC, distD 
    elif  a == 2.0 and 0  in li:
        z = 0.0001
        distA =  abs(i-min_x+z) , abs(j -min_y+z)                  
        distB =  abs(i-min_x+z)  ,  abs(j-max_y+z)                
        distC =  abs(i-max_x+z),  abs(j -min_y+z)                   
        distD =   abs(i-max_x+z)  , abs(j-max_y+z)
        return distA, distB, distC, distD 
    elif a ==1.0:
        distA =  abs(j -min_y)                
        distB =  abs(j-max_y)   
        return distA, distB
    elif a ==1.1:          
        distA =  abs(i-min_x)                  
        distB =  abs(i-max_x) 
        return distA, distB  
    
image = cv2.imread("png_1.png", cv2.IMREAD_UNCHANGED)

clahe ,output_image = clahe_apply(image , (4,4))
#print(f'after clahe:{output_image.max(),output_image.min(),output_image.sum(),output_image.size}')
blurred = cv2.GaussianBlur(image,(5,5),2)
output_image = (output_image * (image/(blurred))**0.3)
end = time.time()
print(f"Execution time: {(end-start)*10**3} ms")
cv2.imwrite('png_1_enhanced.png',output_image)
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(output_image, cmap='gray')
plt.axis('off')
plt.show()
