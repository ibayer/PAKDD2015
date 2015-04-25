library(ggplot2)
library(grid)
save_plots = T

cls <- c(ratio="numeric", mean_acc="numeric", args="numeric", std_acc="numeric")
df = read.csv("../results/agg_results.csv", colClasses=cls, stringsAsFactors=FALSE)
df$exp = paste(df$features, df$args)

# set base acc
df$base_acc = 0
df$base_acc[df$dataset == 'imdb_all'] = .57
df$base_acc[df$dataset == 'citeseer'] = .21
df$base_acc[df$dataset == 'cora'] = .30

df$dataset[df$dataset == 'imdb_all'] = 'IMDb'
df$dataset[df$dataset == 'citeseer'] = 'Citeseer'
df$dataset[df$dataset == 'cora'] = 'Cora'


print(unique(df$features))
print(unique(df$dataset))

theme_set(theme_bw(10))

# bow
i_bow = df$dataset %in% c("Citeseer", "Cora")
i_features = df$features %in% c("bow", "bow_counts", "alchemy_bow_link")
i_count2 = df$features == "counts" & df$args ==2
df_bow = df[i_bow & (i_features | i_count2),]
df_bow$args = as.character(df_bow$args)
df_bow$features[df_bow$features == 'bow'] = 'bow'
df_bow$features[df_bow$features == 'counts'] = 'NCC'
df_bow$features[df_bow$features == 'bow_counts'] = 'bow & NCC'
df_bow$features[df_bow$features == 'alchemy_bow_link'] = 'MLN'

p <- ggplot(df_bow,
            aes(x=ratio, y=mean_acc, group=features, color=features, label=args)) +
    geom_line(aes(linetype=features)) + geom_point(aes(shape=features), size=2) +
    ylab("Average Accuracy") + facet_wrap( ~ dataset, scales="free") +
    scale_x_continuous(breaks = c(1,3,5,7,9)*.1, labels=paste(c(1,3,5,7,9)*10, "%",sep="")) +
    xlab(NULL) + theme(strip.background = element_blank(), legend.position = "bottom",
                       legend.title=element_blank(),
                       legend.key = element_rect(fill = "white", color = "white"),
                       legend.text = element_text(size=10),
                       plot.margin = unit(rep(0, 4), "lines"))

p
if (save_plots)
    ggsave(p, file="../fig/bow.pdf", width=4, height=2.5)


# rwr
i_paper = df$dataset %in% c("Citeseer", "Cora", "IMDb")
i_features = df$features ==  "rwr"
df_depth = df[i_features & i_paper,]
df_depth$args = as.character(df_depth$args)
df_depth$args[df_depth$features == "cluster"&df_depth$args == "0"] = "*"
p <- ggplot(df_depth,
            aes(x=ratio, y=mean_acc, group=exp, color=args, label=args))
p = p + geom_line() + facet_wrap( ~ dataset, scales="free") +
    geom_line(aes(linetype=args)) + geom_point(aes(shape=args), size=2) +
    ylab("Average Accuracy") +
    scale_x_continuous(breaks = c(1,3,5,7,9)*.1, labels=paste(c(1,3,5,7,9)*10, "%",sep="")) +
    xlab(NULL) + theme(strip.background = element_blank(), legend.position = "bottom",
                       legend.title=element_blank(),
                       legend.key = element_rect(fill = "white", color = "white"),
                       legend.text = element_text(size=12),
                       plot.margin = unit(rep(0, 4), "lines"))
p
if (save_plots)
    ggsave(p, file="../fig/rwr.pdf", width=8, height=3)

# cluster
i_paper = df$dataset %in% c("Citeseer", "Cora", "IMDb")
i_features = df$features == "cluster"
i_clusters = df$args %in% c(0, 4, 64, 512, 1024)
df_depth = df[i_features & i_paper & i_clusters,]
df_depth$args = factor(as.numeric(df_depth$args),
                       levels=c(0, 4, 64, 512, 1024),
                       labels=c("all clusterings combined", "4", "64", "512", "1024"))
p <- ggplot(df_depth,
            aes(x=ratio, y=mean_acc, group=exp, color=args, label=args))
p = p + geom_line(aes(linetype=args)) + geom_point(aes(shape=args), size=2) +
    facet_wrap( ~ dataset, scales="free") +
    ylab("Average Accuracy") +
    scale_x_continuous(breaks = c(1,3,5,7,9)*.1, labels=paste(c(1,3,5,7,9)*10, "%",sep="")) +
    xlab(NULL) + theme(strip.background = element_blank(), legend.position = "bottom",
                       legend.title=element_blank(),
                       legend.key = element_rect(fill = "white", color = "white"),
                       legend.text = element_text(size=12),
                       plot.margin = unit(rep(0, 4), "lines"))

p
if (save_plots)
    ggsave(p, file="../fig/cluster.pdf", width=8, height=3)

# depth 
i_paper = df$dataset %in% c("Citeseer", "Cora", "IMDb")
i_features = df$features %in% c("counts", "proba", "ids")
df_depth = df[i_features & i_paper,]
df_depth$args = as.character(df_depth$args)
df_depth$args[df_depth$args == '0'] = "distance 1"
df_depth$args[df_depth$args == '1'] = "distance 1, 2"
df_depth$args[df_depth$args == '2'] = "distance 1, 2, 3"
df_depth$features[df_depth$features == 'ids'] = "neighbor"
df_depth$features[df_depth$features == 'counts'] = "neighbor class counts"
df_depth$features[df_depth$features == 'proba'] = "neighbor class probabilities"
p <- ggplot(df_depth,
            aes(x=ratio, y=mean_acc, group=exp, color=args, label=args))
p = p + geom_line(aes(linetype=args)) + facet_grid(dataset ~ features, scales="free") +
        geom_point(aes(shape=args), size=2) +
    ylab("Average Accuracy") +
    scale_x_continuous(breaks = c(1,3,5,7,9)*.1, labels=paste(c(1,3,5,7,9)*10, "%",sep="")) +
    xlab(NULL) + theme(strip.background = element_blank(), legend.position = "bottom",
                       legend.title=element_blank(),
                       legend.key = element_rect(fill = "white", color = "white"),
                       legend.text = element_text(size=14),
                       plot.margin = unit(rep(0, 4), "lines"))

p
if (save_plots)
    ggsave(p, file="../fig/depth.pdf", width=8, height=7)

# best off
i_features = df$features %in% c("wvrn", "nlb", "id_proba")
i_rwr9 = df$features == 'rwr' & df$args=="0.9"
df_n_only = df[(i_rwr9 |i_features) & i_paper,]
df_n_only$features[df_n_only$features == 'id_proba'] = 'relational features (class counts / probabilities)'
df_n_only$features[df_n_only$features == 'rwr'] = 'relational features (rwr)'
df_n_only$features[df_n_only$features == 'wvrn'] = 'wvRN'
df_n_only$features[df_n_only$features == 'nlb'] = 'nLB'
p <- ggplot(df_n_only,
            aes(x=ratio, y=mean_acc, group=exp, color=features)) +
    geom_line(aes(linetype=features)) + facet_wrap( ~ dataset,scales="free") +
    geom_point(aes(shape=features), size=2)+
    ylab("Average Accuracy") +
    scale_x_continuous(breaks = c(1,3,5,7,9)*.1, labels=paste(c(1,3,5,7,9)*10, "%",sep="")) +
    xlab(NULL) + theme(strip.background = element_blank(), legend.position = "bottom",
                       legend.title=element_blank(),
                       legend.key = element_rect(fill = "white", color = "white"),
                       legend.text = element_text(size=12),
                       plot.margin = unit(rep(0, 4), "lines"))+
    geom_hline(aes(yintercept=base_acc), linetype="dashed")

p
if (save_plots)
    ggsave(p, file="../fig/best_off.pdf", width=8, height=3)


# do unlabeled neighbors matter?
df_unlabeled = df
df_unlabeled$args = as.character(df_unlabeled$args)
i_bow = df$dataset %in% c("Citeseer", "Cora", "IMDb")
i_ids = df$features %in% c("ids_labeled", "ids")
i_not3 = df$args != 3
df_unlabeled$args[df_unlabeled$args == '0'] = "distance 1"
df_unlabeled$args[df_unlabeled$args == '1'] = "distance 1, 2"
df_unlabeled$args[df_unlabeled$args == '2'] = "distance 1, 2, 3"
df_unlabeled$features[df_unlabeled$features == 'ids_labeled'] = "only labeled neighbors"
df_unlabeled$features[df_unlabeled$features == 'ids'] = "all neighbors"
p <- ggplot(df_unlabeled[i_bow & i_ids & i_not3,],
            aes(x=ratio, y=mean_acc, group=exp, color=features, label=args))
#p = p + geom_line() + facet_wrap( ~ dataset,scales="free")+ geom_text(size=4)
 p = p + geom_line(aes(linetype=args)) + facet_wrap( ~ dataset,scales="free") +
    geom_point(aes(shape=args), size=2)+
    ylab("Average Accuracy") +
    scale_x_continuous(breaks = c(1,3,5,7,9)*.1, labels=paste(c(1,3,5,7,9)*10, "%",sep="")) +
    xlab(NULL) + theme(strip.background = element_blank(), legend.position = "bottom",
                       legend.title=element_blank(),
                       legend.key = element_rect(fill = "white", color = "white"),
                       legend.text = element_text(size=12),
                       plot.margin = unit(rep(0, 4), "lines"))

p
if (save_plots)
    ggsave(p, file="../fig/only_labeled.pdf", width=8, height=3.3)
