from umap import umap_ as UMAP
import os, bz2, pickle
from pandas import DataFrame

from bokeh.palettes import Category10
from bokeh.plotting import figure, show, save
from bokeh.io import output_file

from bokeh.embed import file_html
from bokeh.resources import CDN
import wandb

color_hues = Category10[10] * 10
marker_bases = ["circle", "diamond", "inverted_triangle", "square", "star", "triangle"] * 100


# ======================================================================================================================
#  UMapper Class
# ======================================================================================================================
def load_UMapper (full_path):
    ifile = bz2.BZ2File(full_path, 'rb')
    umapper = pickle.load(ifile)
    ifile.close()
    return umapper

class UMapper:
    def __init__(self, identifier_, embedding_dims=2, metric="cosine", n_neighbors=20, ):
        self.identifier = identifier_
        self.embedding_dims = embedding_dims
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.mapper = UMAP(metric=self.metric, n_neighbors=self.n_neighbors)
        self.mapped_embeddings = None
        self.tags = None

    def fit(self, data, tags_=None):
        """data is an array of vectors of shape (n_samples, n_features)
        tags (if available) is an array of strings of shape (n_samples, )"""

        if tags_ is not None:
            self.tags = tags_
        df = DataFrame(data)
        self.mapped_embeddings = self.mapper.fit_transform(df)

    def decode(self, embedding_):
        assert self.mapped_embeddings is not None, "UMapper has not been fitted yet"
        return self.mapper.inverse_transform(embedding_)

    def plot(self, path=None, fname="umap", show_plot=False, save_plot=True, prepare_for_wandb=False):
        assert self.mapped_embeddings is not None, "UMapper has not been fitted yet"
        assert self.embedding_dims == 2, "UMapper can only plot 2D embeddings"

        if path is None:
            path = os.path.join("misc")

        if not os.path.exists(path):
            os.makedirs(path)

        if not fname.endswith(".html"):
            fname += ".html"

        output_file(os.path.join(path, f"{self.identifier}_{fname}"))

        TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

        p = figure(tools=TOOLS)

        tags = self.tags if self.tags is not None else ["NA"] * self.mapped_embeddings.shape[0]

        # get unique tags sorted by frequency
        unique_tags = sorted(set(tags), key=lambda x: tags.count(x), reverse=True)
        hues = color_hues[:len(unique_tags)]
        markers = marker_bases[:len(unique_tags)]
        tag_hue_map = {f"{tag}": hues[i] for i, tag in enumerate(unique_tags)}
        tag_marker_map = {f"{tag}": markers[i] for i, tag in enumerate(unique_tags)}

        data_dict = {}
        for ix, (dim_0, dim_1) in enumerate(self.mapped_embeddings):
            if not tags[ix] in data_dict:
                data_dict.update({tags[ix]: ([], [])})
            data_dict[tags[ix]][0].append(dim_0)
            data_dict[tags[ix]][1].append(dim_1)

        for unique_tag, (dim_0, dim_1) in data_dict.items():
            p.scatter(dim_0, dim_1,
                      fill_color=tag_hue_map[unique_tag],
                      line_color=None, legend_label=unique_tag, marker=tag_marker_map[unique_tag], size=12)

        p.legend.click_policy = "hide"

        if save_plot:
            save(p)

        if show_plot:
            show(p)

        return p if not prepare_for_wandb else wandb.Html(file_html(p, CDN, f"{self.identifier}_{fname}"))

    def dump(self, path=None, fname="evaluator"):  # todo implement in comparator

        if path is None:
            path = os.path.join("misc")

        if not os.path.exists(path):
            os.makedirs(path)

        if not fname.endswith(".Eval.bz2"):
            fname += ".Eval.bz2"

        ofile = bz2.BZ2File(os.path.join(path, f"{self.identifier}_{fname}"), 'wb')
        pickle.dump(self, ofile)
        ofile.close()

