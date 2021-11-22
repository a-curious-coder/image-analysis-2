<h1> Image Analysis Group Coursework 2 </h1>

This is assignment 2/2, image compression, for the Image Analysis module. Docs contained below.  
- [Assignment Document](https://docs.google.com/document/d/1W5qSkQbo6SWXslawJb5RiTPZoJx0WMI86_ApNOPFLrI/edit?usp=sharing)
<!-- Note: Please don't upload the example images to gitlab as they are quite sizable. -->
<h2>Contents</h2>
1.  [Requirements for the assignment](#assrequirements)
2.  [Project Requirements](#projectrequirements)
</br >
<h2>Requirements for the assignment<a name="assrequirements"></a></h2>

- [ ] Load and display original image
- [ ] Save compressed image to memory
- [ ] Read compressed image from memory
- [ ] Display decompressed image
- [ ] Compute and display compression ratio

<h2>Project requirements<a name="projectrequirements"></a></h2>

```py
# How to compile and run project files here
pip install -r requirements.txt
```

```py
# How to compress image
python3 compressImage.py <image source> <image save destination> <compression technique flag>
```

```py
# How to decompress image
python3 compressImage.py <image source> <image save destination> -d <compression technique flag>
```

```py
# How to compare two images
python3 compareImage.py <image 1> <image 2> [v]
```
