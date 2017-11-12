import numpy as np
import cv2

def TemplateMatching(src, temp, stepsize): # src: source image, temp: template image, stepsize: the step size for sliding the template
    mean_t = 0
    var_t = 0
    location = [0, 0]
    # Calculate the mean and variance of template pixel values
    # ------------------ Put your code below ------------------ 
    
    #mean_t = np.mean(temp)
    #var_t = np.var(temp)
    s = 0
    for a in range(0,temp.shape[0]):
        for b in range(0,temp.shape[1]):
            s = s + temp[a,b]
    mean_t = s/(temp.shape[0]*temp.shape[1])
    
    s = 0
    for a in range(0,temp.shape[0]):
        for b in range(0, temp.shape[1]):
            s = s + (temp[a,b]-mean_t)**2
    var_t = s/(temp.shape[0]*temp.shape[1])
            
    
    sum_fenzi = 0
    
    max_corr = 0
    # Slide window in source image and find the maximum correlation
    for i in np.arange(0, src.shape[0] - temp.shape[0], stepsize):
        for j in np.arange(0, src.shape[1] - temp.shape[1], stepsize):
            mean_s = 0
            var_s = 0
            corr = 0
            # Calculate the mean and variance of source image pixel values inside window
            # ------------------ Put your code below ------------------ 
            
            #mean_s = np.mean(src[i:i+temp.shape[0], j:j+temp.shape[1]])
            #var_s = np.var(src[i:i+temp.shape[0], j:j+temp.shape[1]])
            s = 0
            for a in range(i,i+temp.shape[0]):
                for b in range(j,j+temp.shape[1]):
                    s = s + src[a,b]
            mean_s = s/(src.shape[0]*src.shape[1])
            
            s = 0
            for a in range(i, i+temp.shape[0]):
                for b in range(j, j+temp.shape[1]):
                    s = s + (src[a,b]-mean_s)**2
            var_s = s/(temp.shape[0]*temp.shape[1])
                    
            
            # Calculate normalized correlation coefficient (NCC) between source and template
            # ------------------ Put your code below ------------------ 
            for m in range(0, temp.shape[0]):
                for n in range(0, temp.shape[1]):
                    sum_fenzi = sum_fenzi + (src[i+m,j+n] - mean_s)*(temp[m,n] - mean_t)
            corr = sum_fenzi/(temp.shape[0]*temp.shape[1]*var_t*var_s)
            
            if corr > max_corr:
                max_corr = corr
                location = [i, j]
    return location

# load source and template images
source_img = cv2.imread('source_img.jpg',0) # read image in grayscale
temp = cv2.imread('template_img.jpg',0) # read image in grayscale
location = TemplateMatching(source_img, temp, 20);
print(location)
match_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)

# Draw a red rectangle on match_img to show the template matching result
# ------------------ Put your code below ------------------ 

match_img = cv2.rectangle(match_img, (location[0], location[1]), (location[0]+temp.shape[0], location[1]+temp.shape[1]),(255,0,0),3)

# Save the template matching result image (match_img)
# ------------------ Put your code below ------------------ 

cv2.imwrite('match_img.jpg', match_img)
# Display the template image and the matching result
cv2.namedWindow('TemplateImage', cv2.WINDOW_NORMAL)
cv2.namedWindow('MyTemplateMatching', cv2.WINDOW_NORMAL)
cv2.imshow('TemplateImage', temp)
cv2.imshow('MyTemplateMatching', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
