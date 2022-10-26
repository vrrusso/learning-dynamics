run:
	@docker run --gpus all -v "$$PWD:/src" --ipc=host --name $(NAME) -d $(DOCKER-IMAGE) $(CMD)

show:
	@docker logs $(NAME)

remove:
	@docker rm $(NAME)
