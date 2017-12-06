/* 查询因子库中所有因子 */
select * from factor_info;

/* 查询标签库中所有标签 */
select * from tag_info;

/* 查询标签库当前已经打了标签的因子 */
select distinct(factor_info.factor_id), factor_name from factor_info inner join relations on relations.factor_id=factor_info.factor_id;

/* 查询因子的所有标签（以及标签所属类型） */
select * from factor_tag;