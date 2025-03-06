.DEFAULT_GOAL 					:= help
DIR   							:= $(dir $(lastword $(MAKEFILE_LIST)))

include $(DIR)/ai/qwen/Makefile