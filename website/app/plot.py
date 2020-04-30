import io
import matplotlib.pyplot as plt

def plot_alignment(alignment):

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'

    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
        
    #ref https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image
    
    buf = io.BytesIO() #bytes())
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf
